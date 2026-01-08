import argparse
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def preprocess(img_path):
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)

# ================= SMART MODEL LOADER =================
def load_model(model_path, num_classes):
    state_dict = torch.load(model_path, map_location="cpu")

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features

    # 🔥 AUTO-DETECT FC TYPE
    if any(k.startswith("fc.0") for k in state_dict.keys()):
        # Sequential FC head
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, num_classes)
        )
    else:
        # Simple Linear FC
        model.fc = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(state_dict)
    model.eval()
    return model

# ================= LABELS =================
LABELS = {
    "liver": ["Normal", "Abnormal"],
    "tumor": ["Benign", "Malignant"],
    "thyroid": ["Benign", "Malignant"],
    "fatty_liver": ["Mild", "Moderate", "Severe"]
}

MODEL_PATHS = {
    "liver": MODELS_DIR / "liver/liver_ultrasound_resnet18.pth",
    "tumor": MODELS_DIR / "liver/tumor_model_ft.pth",
    "thyroid": MODELS_DIR / "thyroid/thyroid_model_finetuned.pth",
    "fatty_liver": MODELS_DIR / "fatty_liver/fatty_liver_severity_model_ft.pth"
}

NUM_CLASSES = {
    "liver": 2,
    "tumor": 2,
    "thyroid": 2,
    "fatty_liver": 3
}

# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--organ",
        required=True,
        choices=["liver", "tumor", "thyroid", "fatty_liver"]
    )
    args = parser.parse_args()

    model = load_model(
        MODEL_PATHS[args.organ],
        NUM_CLASSES[args.organ]
    )

    image_tensor = preprocess(args.image)

    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        conf, idx = torch.max(probs, dim=1)

    confidence = min(conf.item(), 0.95)

    print("\n========== PREDICTION ==========")
    print(f"Organ      : {args.organ}")
    print(f"Prediction : {LABELS[args.organ][idx.item()]}")
    print(f"Confidence : {confidence * 100:.2f}%")
    print("================================\n")

if __name__ == "__main__":
    main()
