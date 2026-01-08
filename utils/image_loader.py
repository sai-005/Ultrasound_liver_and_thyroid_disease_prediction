import cv2
import torch
from torchvision import transforms
from PIL import Image

def load_ultrasound_image(img_path, img_size=224):
    image = Image.open(img_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = transform(image)
    image = image.unsqueeze(0)  # add batch dim
    return image
