import os
import json
import time
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix

from common import IMG, MEAN, STD, eval_tf, build_resnet, gradcam

torch.manual_seed(42)
np.random.seed(42)
os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------------ 1. Data
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)),   # augmentation: random zoom and crop
    transforms.RandomHorizontalFlip(),                      # augmentation: mirror
    transforms.ColorJitter(0.2, 0.2, 0.2),                  # augmentation: lighting changes
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

ds_train_aug = datasets.OxfordIIITPet("data", split="trainval", target_types="category",
                                      transform=train_tf, download=True)
ds_train_plain = datasets.OxfordIIITPet("data", split="trainval", target_types="category",
                                        transform=eval_tf, download=True)
test_ds = datasets.OxfordIIITPet("data", split="test", target_types="category",
                                 transform=eval_tf, download=True)
classes = ds_train_aug.classes
n_classes = len(classes)

# train / validation split of the "trainval" part (80 / 20, stratified by breed)
labels_tv = np.array(ds_train_aug._labels)
tr_idx, va_idx = train_test_split(np.arange(len(labels_tv)), test_size=0.2,
                                  stratify=labels_tv, random_state=42)
train_ds = Subset(ds_train_aug, tr_idx)    # with augmentation
val_ds = Subset(ds_train_plain, va_idx)    # no augmentation

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
print(f"Classes: {n_classes} | Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

criterion = nn.CrossEntropyLoss()


# ------------------------------------------------------------ 2. Train / eval helpers
def run_epoch(model, loader, optimizer=None, frozen=()):
    training = optimizer is not None
    model.train(training)
    if training:
        for m in frozen:      # keep frozen layers (and their BatchNorm statistics) in eval mode
            m.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def fit(name, model, optimizer, scheduler, epochs, frozen=()):
    print(f"\n=== Training: {name} ===")
    best_acc, best_state, hist = 0.0, None, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, optimizer, frozen)
        vl, va = run_epoch(model, val_loader)
        if scheduler is not None:
            scheduler.step()
        hist.append({"epoch": ep, "train_acc": ta, "val_acc": va, "train_loss": tl, "val_loss": vl})
        print(f"epoch {ep}/{epochs} | train acc {ta:.3f} | val acc {va:.3f} | "
              f"train loss {tl:.3f} | val loss {vl:.3f} | {time.time() - t0:.0f}s")
        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)    # keep the epoch with the best validation accuracy
    return pd.DataFrame(hist), best_acc


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        probs.append(F.softmax(model(x.to(device)), dim=1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def make_resnet(trainable):
    m = build_resnet(n_classes, pretrained=True)
    for p in m.parameters():
        p.requires_grad = False
    for name in trainable:
        for p in getattr(m, name).parameters():
            p.requires_grad = True
    all_blocks = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
    frozen = [getattr(m, n) for n in all_blocks if n not in trainable]
    return m.to(device), frozen


class SmallCNN(nn.Module):
    def __init__(self, n):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128), block(128, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(128, n))

    def forward(self, x):
        return self.head(self.features(x))


# ------------------------------------------------------------------ 3. Train 3 models
histories, val_best = {}, {}

# Model 1: small CNN from scratch
m1 = SmallCNN(n_classes).to(device)
opt = torch.optim.Adam(m1.parameters(), lr=1e-3)
histories["CNN from scratch"], val_best["CNN from scratch"] = fit("CNN from scratch", m1, opt, None, 8)

# Model 2: pretrained ResNet18, frozen backbone, train only the new head
m2, frozen2 = make_resnet(["fc"])
opt = torch.optim.Adam(m2.fc.parameters(), lr=1e-3)
histories["ResNet18 frozen"], val_best["ResNet18 frozen"] = fit(
    "ResNet18 feature extraction", m2, opt, None, 5, frozen2)

# Model 3: pretrained ResNet18, fine-tune the last block + head (different learning rates)
m3, frozen3 = make_resnet(["layer4", "fc"])
opt = torch.optim.AdamW([
    {"params": m3.layer4.parameters(), "lr": 1e-4},
    {"params": m3.fc.parameters(), "lr": 1e-3},
], weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
histories["ResNet18 fine-tuned"], val_best["ResNet18 fine-tuned"] = fit(
    "ResNet18 fine-tuning", m3, opt, sched, 6, frozen3)

trained = {"CNN from scratch": m1, "ResNet18 frozen": m2, "ResNet18 fine-tuned": m3}

# ------------------------------------------------------ 4. Final evaluation on TEST (once)
results, test_preds = {}, {}
for name, model in trained.items():
    probs, y = predict(model, test_loader)
    preds = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    results[name] = {
        "val_acc": val_best[name],
        "test_acc": float((preds == y).mean()),
        "test_top5": float((top5 == y[:, None]).any(axis=1).mean()),
        "test_macro_f1": float(f1_score(y, preds, average="macro")),
    }
    test_preds[name] = preds

res = pd.DataFrame(results).T
print("\nTEST RESULTS:")
print(res.round(4).to_string())

# training curves
plt.figure(figsize=(7, 4.5))
for name, h in histories.items():
    plt.plot(h["epoch"], h["val_acc"], marker="o", label=f"{name} (val)")
    plt.plot(h["epoch"], h["train_acc"], linestyle="--", alpha=0.5, label=f"{name} (train)")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training curves")
plt.legend(fontsize=7)
plt.tight_layout()
plt.savefig("outputs/training_curves.png")
plt.close()

res["test_acc"].sort_values().plot(kind="barh", title="Test accuracy by model")
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# ----------------------------------------------------------- 5. Error analysis (best model)
ft = m3
preds = test_preds["ResNet18 fine-tuned"]
y = np.array(test_ds._labels)
cm = confusion_matrix(y, preds)
off = cm.copy()
np.fill_diagonal(off, 0)
top_pairs = np.dstack(np.unravel_index(np.argsort(-off.ravel())[:5], off.shape))[0]
recall = cm.diagonal() / cm.sum(axis=1)
worst = np.argsort(recall)[:5]

lines = ["Most confused breeds (true -> predicted : count):"]
for t, p in top_pairs:
    lines.append(f"  {classes[t]} -> {classes[p]} : {off[t, p]}")
lines.append("\nWeakest breeds (recall):")
for c in worst:
    lines.append(f"  {classes[c]} : {recall[c]:.2f}")
print("\n" + "\n".join(lines))
with open("outputs/error_analysis.txt", "w") as f:
    f.write("\n".join(lines))

# ----------------------------------------------------------------- 6. Grad-CAM grid
def denorm(t):
    img = t.cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)


rng = np.random.default_rng(0)
right = rng.choice(np.where(preds == y)[0], 4, replace=False)
wrong = rng.choice(np.where(preds != y)[0], 4, replace=False)
fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
for ax, idx in zip(axes.ravel(), list(right) + list(wrong)):
    x, _ = test_ds[int(idx)]
    cam, p = gradcam(ft, x.unsqueeze(0).to(device))
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    ax.imshow(0.6 * denorm(x) + 0.4 * heat)
    ax.set_title(f"True: {classes[y[idx]]}\nPred: {classes[p.argmax()]} ({p.max():.0%})", fontsize=8)
    ax.axis("off")
plt.suptitle("Grad-CAM: top row correct, bottom row wrong")
plt.tight_layout()
plt.savefig("outputs/gradcam.png")
plt.close()

# ----------------------------------------------------------------- 7. Save for the app
state = {k: v.cpu() for k, v in ft.state_dict().items()}
torch.save({"state_dict": state, "classes": classes}, "outputs/best_model.pt")
with open("outputs/metrics.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDone! Model and plots saved in outputs/")