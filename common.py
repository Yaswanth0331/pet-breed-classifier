import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

IMG = 160
MEAN = [0.485, 0.456, 0.406]   # ImageNet statistics, required for pretrained models
STD = [0.229, 0.224, 0.225]

eval_tf = transforms.Compose([
    transforms.Resize(176),
    transforms.CenterCrop(IMG),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def build_resnet(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    m.fc = nn.Linear(m.fc.in_features, num_classes)   # new classification head
    return m


def gradcam(model, x):
    """x has shape (1, 3, H, W). Returns (heatmap in [0, 1] as an HxW array, class probabilities)."""
    model.eval()
    store = {}

    def fwd_hook(module, inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.update(grad=g))

    handle = model.layer4.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            for p in model.parameters():
                p.requires_grad_(True)
            model.zero_grad()
            out = model(x)
            cls = out.argmax(1).item()
            out[0, cls].backward()
    finally:
        handle.remove()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)          # importance of each channel
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))  # weighted sum of feature maps
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    probs = F.softmax(out.detach(), dim=1)[0].cpu().numpy()
    return cam, probs