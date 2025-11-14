import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MapillaryDataset, MapillaryDatasetAug
from advanced_model import DeepLabV3Plus
import configs.config as cfg
import numpy as np


def compute_class_weights(dataset, num_classes=124):
    counts = np.zeros(num_classes, dtype=np.int64)
    n = min(len(dataset), 200) 
    for i in range(n):
        _, mask = dataset[i]
        mask_np = mask.numpy().flatten()
        for cls in range(num_classes):
            counts[cls] += np.sum(mask_np == cls)
    freq = counts / counts.sum()
    c = 1.02
    weights = 1.0 / np.log(c + freq)
    weights = torch.from_numpy(weights).float()
    return weights

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    train_ds = MapillaryDatasetAug(
        root_dir=cfg.ROOT_DIR, split=cfg.TRAIN_SPLIT, patch_size=cfg.PATCH_SIZE,
        flip_prob=cfg.FLIP_PROB, rotation_deg=cfg.ROTATION_DEGREES,
        color_jitter_params=cfg.COLOR_JITTER
    )
    val_ds = MapillaryDataset(root_dir=cfg.ROOT_DIR, split=cfg.VAL_SPLIT, patch_size=cfg.PATCH_SIZE)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    model = DeepLabV3Plus(
        num_classes=cfg.NUM_CLASSES,
        layers=[3,4,6,3],
        num_groups=32,
        use_attention=True
    )
    model.to(device)

    class_weights = compute_class_weights(train_ds, num_classes=cfg.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)

    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE,
                                momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)

    best_val_loss = float('inf')
    for epoch in range(1, cfg.NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % 50 == 0:
                avg = running_loss / 50
                print(f"[Epoch {epoch}] Step {i+1}/{len(train_loader)}, Loss: {avg:.4f}")
                running_loss = 0.0
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(cfg.OUTPUT_DIR, 'best_advanced_model.pth')
            torch.save(model.state_dict(), path)
            print(f"New best model saved: {path} (val_loss={val_loss:.4f})")

    final_path = os.path.join(cfg.OUTPUT_DIR, f'last_advanced_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Training completed. Final model saved: {final_path}")

if __name__ == '__main__':
    train()

