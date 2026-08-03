"""多进程并行相机渲染 — 绕过 EGL 单线程限制。

⚠️ WSL 兼容性说明:
  此模块在 WSL + MESA D3D12 下不可用 —— D3D12 设备是进程单例，
  fork 后子进程无法创建新的 D3D12 设备（D3D12: Removing Device.）。
  在原生 Linux + EGL 下应该可以工作。

每个相机在独立进程中持有自己的 EGL context + MuJoCo Renderer，
主进程通过 shared_memory 传递 mjData 状态，并行渲染后回传图像。

同步协议（每 worker 两个 Event）：
  trigger_event:  主进程 set → worker 开始渲染 → worker clear
  complete_event: worker 渲染完成 set → 主进程读取结果 → 主进程 clear

Usage:
    pool = RenderProcessPool(scene_path, camera_specs)
    pool.start()
    images = pool.render_all(qpos, qvel)  # 并行渲染所有相机
    pool.stop()
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from dataclasses import dataclass
from multiprocessing import Process, Event
from multiprocessing import shared_memory as sm

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
_log = logging.getLogger(__name__)

# 最大关节数（UR5e×2 + free joints ≈ 32，留足余量）
_MAX_NQ = 128
_MAX_NV = 128


@dataclass
class CameraSpec:
    """相机规格（可 pickle，用于跨进程传递）。"""
    name: str
    height: int
    width: int


# ═══════════════════════════════════════════════════════
# Worker 进程
# ═══════════════════════════════════════════════════════

def _worker_main(
    scene_path: str,
    cam_spec: CameraSpec,
    shm_name_state: str,
    shm_name_rgb: str,
    shm_name_depth: str,
    trigger: Event,
    complete: Event,
    shutdown: Event,
) -> None:
    """Worker 进程 — 持有独立 EGL context 和 MuJoCo Renderer。"""
    import mujoco

    shm_state = sm.SharedMemory(name=shm_name_state)
    shm_rgb = sm.SharedMemory(name=shm_name_rgb)
    shm_depth = sm.SharedMemory(name=shm_name_depth)

    H, W = cam_spec.height, cam_spec.width
    nq = shm_state.size // 8 // 2

    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)
    cam_id = model.camera(cam_spec.name).id
    renderer = mujoco.Renderer(model, height=H, width=W)

    # numpy 视图 → 共享内存
    state_buf = np.ndarray(shm_state.size, dtype=np.uint8, buffer=shm_state.buf)
    rgb_buf = np.ndarray((H, W, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    depth_buf = np.ndarray((H, W), dtype=np.float32, buffer=shm_depth.buf)
    qpos_view = state_buf[:nq * 8].view(np.float64)
    qvel_view = state_buf[nq * 8:2 * nq * 8].view(np.float64)

    try:
        while not shutdown.is_set():
            trigger.wait()
            if shutdown.is_set():
                break
            trigger.clear()

            try:
                data.qpos[:len(qpos_view)] = qpos_view
                data.qvel[:len(qvel_view)] = qvel_view
                mujoco.mj_forward(model, data)

                renderer.update_scene(data, camera=cam_id)
                renderer.disable_depth_rendering()
                rgb_buf[:] = renderer.render()
                renderer.enable_depth_rendering()
                depth_buf[:] = renderer.render()
            except Exception:
                _log.exception("Worker %s render error", cam_spec.name)

            complete.set()
    finally:
        renderer.close()
        shm_state.close()
        shm_rgb.close()
        shm_depth.close()


# ═══════════════════════════════════════════════════════
# 进程池
# ═══════════════════════════════════════════════════════

class RenderProcessPool:
    """多进程相机渲染池。

    每个相机一个独立进程，通过 shared_memory 零拷贝传递数据。
    """

    def __init__(self, scene_path: str, cameras: list[CameraSpec]) -> None:
        self._scene_path = scene_path
        self._cameras = cameras
        self._processes: list[Process] = []
        self._triggers: list[Event] = []
        self._completes: list[Event] = []
        self._shutdowns: list[Event] = []
        self._shms: list[tuple[sm.SharedMemory, sm.SharedMemory, sm.SharedMemory]] = []
        self._started = False
        atexit.register(self.stop)

    # ── 生命周期 ──

    def start(self) -> None:
        if self._started:
            return

        for cam in self._cameras:
            H, W = cam.height, cam.width

            shm_state = sm.SharedMemory(create=True, size=_MAX_NQ * 8 + _MAX_NV * 8)
            shm_rgb = sm.SharedMemory(create=True, size=H * W * 3)
            shm_depth = sm.SharedMemory(create=True, size=H * W * 4)

            trigger = Event()
            complete = Event()
            shutdown = Event()

            p = Process(
                target=_worker_main,
                args=(self._scene_path, cam,
                      shm_state.name, shm_rgb.name, shm_depth.name,
                      trigger, complete, shutdown),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
            self._triggers.append(trigger)
            self._completes.append(complete)
            self._shutdowns.append(shutdown)
            self._shms.append((shm_state, shm_rgb, shm_depth))

        self._started = True
        _log.info("RenderProcessPool: %d workers started", len(self._cameras))

    def stop(self) -> None:
        if not self._started:
            return
        for shutdown in self._shutdowns:
            shutdown.set()
        for trigger in self._triggers:
            trigger.set()
        for p in self._processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        for shm_state, shm_rgb, shm_depth in self._shms:
            for s in (shm_state, shm_rgb, shm_depth):
                s.close()
                try:
                    s.unlink()
                except Exception:
                    pass
        self._processes.clear()
        self._triggers.clear()
        self._completes.clear()
        self._shutdowns.clear()
        self._shms.clear()
        self._started = False

    # ── 渲染 ──

    def render_all(
        self, qpos: np.ndarray, qvel: np.ndarray, timeout: float = 5.0,
    ) -> list[dict[str, np.ndarray]]:
        """并行渲染所有相机。

        Returns:
            [{"rgb": uint8(H,W,3), "depth": float32(H,W)}, ...]
        """
        if not self._started:
            raise RuntimeError("RenderProcessPool not started")

        qpos_arr = np.asarray(qpos, dtype=np.float64).ravel()
        qvel_arr = np.asarray(qvel, dtype=np.float64).ravel()

        # ── 写入状态到所有 worker ──
        for i, (shm_state, _, _) in enumerate(self._shms):
            nq = min(len(qpos_arr), shm_state.size // 8 // 2)
            nv = min(len(qvel_arr), shm_state.size // 8 // 2)
            buf = np.ndarray(shm_state.size, dtype=np.uint8, buffer=shm_state.buf)
            buf[:nq * 8] = qpos_arr[:nq].view(np.uint8)
            buf[nq * 8:nq * 8 + nv * 8] = qvel_arr[:nv].view(np.uint8)

        # ── 触发所有 worker 并行渲染 ──
        for trigger, complete in zip(self._triggers, self._completes):
            complete.clear()
            trigger.set()

        # ── 等待所有 worker 完成，读取结果 ──
        results: list[dict[str, np.ndarray]] = []
        for i, complete in enumerate(self._completes):
            if not complete.wait(timeout=timeout):
                _log.warning("Worker %d (%s) render timeout", i, self._cameras[i].name)
            complete.clear()

            _, shm_rgb, shm_depth = self._shms[i]
            cam = self._cameras[i]

            rgb = np.ndarray(
                (cam.height, cam.width, 3), dtype=np.uint8, buffer=shm_rgb.buf
            ).copy()
            depth = np.ndarray(
                (cam.height, cam.width), dtype=np.float32, buffer=shm_depth.buf
            ).copy()

            results.append({"rgb": rgb, "depth": depth})

        return results

    @property
    def is_started(self) -> bool:
        return self._started
