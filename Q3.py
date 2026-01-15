import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import requests

st.title("Real-Time Image Classification")

labels = requests.get(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
).text.splitlines()

model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])

img_file = st.camera_input("Capture Image")

if img_file:
    img = Image.open(img_file)
    st.image(img, caption="Captured Image")

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)

    top5 = torch.topk(probs, 5)
    st.table({
        "Label": [labels[i] for i in top5.indices],
        "Probability": top5.values.numpy()
    })
