import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- DEVICE ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- TASK CONFIG ----------------
TASKS = {
    "thyroid": {
        "data": "data_thyroid",
        "model": "models/thyroid/thyroid_model_finetuned.pth",
        "classes": ["Benign", "Malignant"],
        "num_classes": 2,
        "head": "mlp"   # trained with Sequential head
    },

    "liver_tumor": {
        "data": "data",
        "model": "models/liver/tumor_model_ft.pth",
        "classes": ["Benign", "Malignant", "Normal"],
        "num_classes": 3,
        "head": "linear"
    },

    "fatty_liver": {
        "data": "data_fatty",
        "model": "models/fatty_liver/fatty_liver_severity_model_ft.pth",
        "classes": ["mild", "moderate", "normal", "severe"],
        "num_classes": 4,
        "head": "linear"
    }
}

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- MODEL LOADER ----------------
def load_model(task_cfg):
    model = models.resnet18(weights=None)

    # 🔹 Thyroid model head (must match training)
    if task_cfg["head"] == "mlp":
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(model.fc.in_features, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, task_cfg["num_classes"])
        )

    # 🔹 Liver tumor & fatty liver heads
    else:
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            task_cfg["num_classes"]
        )

    model.load_state_dict(
        torch.load(task_cfg["model"], map_location=DEVICE)
    )

    model.to(DEVICE)
    model.eval()
    return model

# ---------------- EVALUATION ----------------
def evaluate(task_name, cfg):
    print(f"\n🔍 Evaluating {task_name.upper()} model")

    dataset = datasets.ImageFolder(
        root=cfg["data"],
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False
    )

    model = load_model(cfg)

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(labels.numpy())
            y_pred.extend(preds)

    # ---------- REPORT ----------
    print("\n📊 Classification Report")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=cfg["classes"]
        )
    )

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=cfg["classes"],
        yticklabels=cfg["classes"]
    )
    plt.title(f"{task_name.upper()} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    for task, cfg in TASKS.items():
        evaluate(task, cfg)
