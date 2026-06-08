"""模仿学习训练器。"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from .losses import build_loss


class ImitationTrainer:
    """模仿学习训练器。
    
    训练 actor 网络以模仿专家动作。
    """

    def __init__(
        self,
        actor: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        loss_type: str = "mse",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        state_dropout: float = 0.0,
        visual_dropout: float = 0.0,
    ) -> None:
        """初始化训练器。
        
        Args:
            actor: Actor 网络
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            learning_rate: 学习率
            loss_type: 损失类型
            device: 设备（"cuda" 或 "cpu"）
            state_dropout: 随机丢弃 state 的概率（0~1），强制网络依赖视觉
            visual_dropout: 随机丢弃视觉的概率（0~1），强制网络依赖 state
        """
        self.actor = actor.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = Adam(self.actor.parameters(), lr=learning_rate)
        self.loss_fn = build_loss(loss_type)
        
        # 配置模态 dropout
        if hasattr(self.actor, 'set_modality_dropout'):
            self.actor.set_modality_dropout(
                state_dropout=state_dropout,
                visual_dropout=visual_dropout,
            )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_actor_state = None

    def train_epoch(self) -> float:
        """训练一个 epoch。
        
        Returns:
            平均训练损失
        """
        self.actor.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.train_loader:
            # 移到设备
            camera = batch["camera"].to(self.device)
            actor_state = batch["actor_state"].to(self.device)
            task = batch["task"].to(self.device)
            expert_action = batch["action"].to(self.device)
            
            # 构造观测
            obs = {
                "camera": camera,
                "actor_state": actor_state,
                "task": task,
            }
            
            # 前向传播
            pred_action = self.actor(obs, deterministic=True)
            
            # 计算损失
            loss = self.loss_fn(pred_action, expert_action)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss

    def validate(self) -> float:
        """验证一个 epoch。
        
        Returns:
            平均验证损失
        """
        if self.val_loader is None:
            return float("inf")
        
        self.actor.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                # 移到设备
                camera = batch["camera"].to(self.device)
                actor_state = batch["actor_state"].to(self.device)
                task = batch["task"].to(self.device)
                expert_action = batch["action"].to(self.device)
                
                # 构造观测
                obs = {
                    "camera": camera,
                    "actor_state": actor_state,
                    "task": task,
                }
                
                # 前向传播
                pred_action = self.actor(obs, deterministic=True)
                
                # 计算损失
                loss = self.loss_fn(pred_action, expert_action)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        self.val_losses.append(avg_loss)
        
        # 保存最佳模型
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.best_actor_state = {k: v.cpu().clone() for k, v in self.actor.state_dict().items()}
        
        return avg_loss

    def train(
        self,
        num_epochs: int = 10,
        save_dir: Optional[str | Path] = None,
        start_epoch: int = 0,
        save_every: int = 10,
        print_interval: int = 1,
    ) -> dict:
        """训练 actor 网络。

        Args:
            num_epochs: 训练 epoch 数。
            save_dir: 保存目录（可选）。
            start_epoch: 起始 epoch（用于恢复训练）。
            save_every: 每 N 个 epoch 保存一次权重。
            print_interval: 打印间隔。

        Returns:
            训练统计信息字典。
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training for {num_epochs} epochs (start={start_epoch})...")

        for epoch in range(num_epochs):
            current_epoch = start_epoch + epoch
            train_loss = self.train_epoch()
            val_loss = self.validate()

            if (epoch + 1) % print_interval == 0:
                msg = f"Epoch {current_epoch + 1}/{start_epoch + num_epochs}: train_loss={train_loss:.6f}"
                if self.val_loader is not None:
                    msg += f", val_loss={val_loss:.6f}"
                print(msg)

            # 定期保存
            if save_dir is not None and (current_epoch + 1) % save_every == 0:
                self.save_actor(save_dir / f"checkpoint_epoch_{current_epoch + 1}.pt")
        
        # 恢复最佳模型
        if self.best_actor_state is not None:
            self.actor.load_state_dict(self.best_actor_state)
        
        # 保存最终模型
        if save_dir is not None:
            best_actor_path = save_dir / "best_actor.pt"
            self.save_actor(best_actor_path)
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

    def save_actor(self, path: str | Path) -> None:
        """保存 actor 网络。
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "model_cfg": getattr(self.actor, "model_cfg", None),
        }
        
        torch.save(checkpoint, path)
        print(f"✓ Actor 保存到 {path}")

    def save_checkpoint(self, path: str | Path) -> None:
        """保存完整检查点。
        
        Args:
            path: 保存路径
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        
        torch.save(checkpoint, path)
