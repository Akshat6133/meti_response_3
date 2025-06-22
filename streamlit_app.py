import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same Generator model definition as before
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        img = img.view(-1, 1, 28, 28)
        return img

# Load the trained generator weights
generator = Generator().to(device)
generator.load_state_dict(torch.load("generator.pth", map_location=device))
generator.eval()

st.title("Handwritten Digit Generator (MNIST style)")

digit = st.slider("Select Digit to Generate", 0, 9, 0)

if st.button("Generate 5 Images"):
    # Generate 5 images for the chosen digit
    noise = torch.randn(5, 100, device=device)
    labels = torch.full((5,), digit, dtype=torch.long, device=device)
    with torch.no_grad():
        gen_imgs = generator(noise, labels).cpu()
    
    cols = st.columns(5)
    for i in range(5):
        img = gen_imgs[i].squeeze(0).numpy()
        # Rescale from [-1,1] to [0,255]
        img = ((img + 1) * 127.5).astype(np.uint8)
        im_pil = Image.fromarray(img, mode='L')
        cols[i].image(im_pil, caption=f"Digit: {digit}", use_column_width=True)
