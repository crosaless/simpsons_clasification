import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from utils import load_model, predict_character

model, idx_to_class = load_model('prod/modelo.pth')
model.eval()

# Se hace la misma transformación que en el entrenamiento 
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

st.title("Detector de Personajes de Los Simpsons utilizando pérdida de las trillizas")

uploaded_file = st.file_uploader("Subí una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = predict_character(model, tensor, idx_to_class)

    st.success(f"Personaje detectado: **{prediction}**")
