import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib
from PIL import Image
from torchvision import transforms

from common import IMG, eval_tf, build_resnet, gradcam

st.set_page_config(page_title="Pet Breed Classifier", page_icon="🐶")


@st.cache_resource
def load():
    ckpt = torch.load("outputs/best_model.pt", map_location="cpu")
    model = build_resnet(len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["classes"]


model, classes = load()

st.title("🐶🐱 Pet Breed Classifier")
st.write("Upload a photo of a cat or dog. The model knows 37 breeds and shows where it looked (Grad-CAM).")

file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    x = eval_tf(img).unsqueeze(0)
    cam, probs = gradcam(model, x)

    view = transforms.CenterCrop(IMG)(transforms.Resize(176)(img))
    view = np.asarray(view).astype(np.float32) / 255
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    overlay = 0.6 * view + 0.4 * heat

    c1, c2 = st.columns(2)
    c1.image(view, caption="Model input", use_container_width=True)
    c2.image(overlay, caption="Where the model looked", use_container_width=True, clamp=True)

    top = np.argsort(-probs)[:5]
    df = pd.DataFrame({"breed": [classes[i] for i in top], "probability": probs[top]})
    if probs[top[0]] < 0.5:
        st.warning("Low confidence. This may not be one of the 37 breeds, or may not be a cat or dog.")
    else:
        st.success(f"Prediction: **{df.iloc[0]['breed']}** ({df.iloc[0]['probability']:.0%} confidence)")
    st.bar_chart(df.set_index("breed"))