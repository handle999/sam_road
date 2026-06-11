import os
from datetime import datetime
from lightning.pytorch.callbacks import Callback


class TextLogCallback(Callback):
    """Write train loss/metrics to .txt in real-time."""

    def __init__(self, log_path):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("SAM-Road Training Log - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + chr(10))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        step = batch_idx + 1
        total = len(trainer.train_dataloader)
        parts = [f"Epoch {epoch}: {step}/{total}"]
        for key in ["train_mask_loss", "train_topo_loss", "train_loss"]:
            val = metrics.get(key)
            if val is not None:
                v = val.item() if hasattr(val, "item") else float(val)
                parts.append(f"{key}={v:.4f}")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(", ".join(parts) + chr(10))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch
        parts = [f"Epoch {epoch} val:"]
        for key in ["val_mask_loss", "val_topo_loss", "val_loss", "keypoint_iou", "road_iou", "topo_f1"]:
            val = metrics.get(key)
            if val is not None:
                v = val.item() if hasattr(val, "item") else float(val)
                parts.append(f"{key}={v:.4f}")
        log_line = ", ".join(parts)
        print(f"\n{log_line}")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(log_line + chr(10))

    def on_train_epoch_end(self, trainer, pl_module):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"--- End of Epoch {trainer.current_epoch} ---" + chr(10))


class EarlyStoppingCallback(Callback):
    """Stop training if val_loss does not improve for patience epochs."""

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float("inf")
        self.counter = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is None:
            return
        val_loss = val_loss.item() if hasattr(val_loss, "item") else float(val_loss)
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"[EarlyStopping] val_loss={val_loss:.4f} no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                print(f"[EarlyStopping] Triggered! Best val_loss={self.best_val_loss:.4f}")
                trainer.should_stop = True
