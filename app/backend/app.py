from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
import numpy as np
import cv2

# ================= APP =================
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent
INFO_PATH = BASE_DIR / "info" / "medical_info.json"
PROJECT_ROOT = BASE_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# ================= LOAD RAG =================
with open(INFO_PATH, "r", encoding="utf-8") as f:
    MEDICAL_INFO = json.load(f)

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= MODEL LOADER =================

def load_model(path, classes):
    state = torch.load(path, map_location="cpu")
    model = models.resnet18(weights=None)
    in_f = model.fc.in_features

    # 🔑 Detect how model was trained
    if any(k.startswith("fc.0") for k in state.keys()):
        # Model trained with Sequential head
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_f, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, classes)
        )
    else:
        # Model trained with simple Linear head
        model.fc = torch.nn.Linear(in_f, classes)

    model.load_state_dict(state)
    model.eval()
    return model

MODELS = {
    "liver": load_model(MODELS_DIR / "liver/liver_ultrasound_resnet18.pth", 2),
    "thyroid": load_model(MODELS_DIR / "thyroid/thyroid_model_finetuned.pth", 2),
}

LABELS = {
    "liver": ["normal", "fatty_liver"],
    "thyroid": ["benign", "malignant"],
}

# ================= GRAD-CAM =================
class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        layer = self.model.layer4[-1]
        layer.register_forward_hook(self._fwd)
        layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _, __, out):
        self.activations = out

    def _bwd(self, _, gin, gout):
        self.gradients = gout[0]

    def generate(self, x, idx):
        self.model.zero_grad()
        out = self.model(x)
        out[0, idx].backward()

        acts = self.activations[0].detach().numpy()
        grads = self.gradients[0].detach().numpy()
        weights = grads.mean(axis=(1,2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i,w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam,0)
        cam = cv2.resize(cam,(224,224))
        cam /= cam.max() + 1e-8
        return cam

# ================= HOME =================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Ultrasound AI</title>

<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">

<style>
body{
  margin:0;height:100vh;
  background:radial-gradient(circle at top,#111,#000);
  font-family:Inter;color:white;
  display:flex;align-items:center;justify-content:center;
}

.upload-card{
  width:1100px;
  background:#0b0b0b;
  border-radius:26px;
  padding:40px;
   box-shadow:
    0 0 0 1px rgba(255,106,0,0.4),
    0 0 40px rgba(255,106,0,0.15);
  
}

.upload-title{
  font-family:'Space Grotesk';
  font-size:32px;
  margin-bottom:28px;
}

.upload-row{
  display:grid;
  grid-template-columns:1fr 1fr;
  gap:40px;
}

.upload-left{
  display:flex;
  flex-direction:column;
  gap:18px;
}

.upload-left select{
  padding:14px;
  border-radius:14px;
  border:none;
}

.upload-dropzone{
  border:2px dashed #333;
  border-radius:20px;
  height:180px;
  display:flex;
  align-items:center;
  justify-content:center;
  text-align:center;
  cursor:pointer;
  position:relative;
  border-color:#ff6a00;
  box-shadow:0 0 20px rgba(255,106,0,0.4);
}

.upload-filename{
  color:#ff6a00;
  font-weight:500;
}

.upload-dropzone input{
  display:none;
}

.gradcam-toggle{
  display:flex;
  align-items:center;
  gap:14px;
}

.gradcam-toggle input{
  width:22px;
  height:22px;
  accent-color:#ff6a00;
}

.predict-btn{
  margin-top:10px;
  padding:16px;
  border-radius:999px;
  border:2px solid #ff6a00;
  background:transparent;
  color:#ff6a00;
  font-family:'Space Grotesk';
  cursor:pointer;
}

.predict-btn:hover{
  background:#ff6a00;
  color:black;
}

.upload-preview{
  display:flex;
  align-items:center;
  justify-content:center;
}
.preview-image{
  width:100%;
  border-radius:20px;
  border:2px solid #ff6a00;
  position:absolute;
  inset:0;
  object-fit:contain;
  opacity:0;
  transition:opacity 0.5s ease;
}

.preview-image.show{
  opacity:1;
}

.upload-preview{
  position:relative;
  height:360px;
}

</style>
</head>

<body>

<form class="upload-card" action="/predict" method="post" enctype="multipart/form-data">

  <div class="upload-title">Upload Ultrasound</div>

  <div class="upload-row">

    <!-- LEFT -->
    <div class="upload-left">

      <label>Select Organ</label>
      <select name="organ" required>
        <option value="liver">Liver</option>
        <option value="thyroid">Thyroid</option>
      </select>

      <label class="upload-dropzone" id="dropZone">
        <input type="file" name="image" id="fileInput" accept="image/*" required>
        <div id="dropText">
          Choose file<br>or drag image here
        </div>
      </label>

      <div class="gradcam-toggle">
        <input type="checkbox" name="gradcam" value="on">
        <span>Enable Grad-CAM</span>
      </div>

      <button type="submit" class="predict-btn">Predict</button>

    </div>

    <!-- RIGHT -->
<div class="upload-preview">
  <!-- Dummy image -->
  <img src="/static/placeholder.png"
       id="dummyGradcam"
       class="preview-image show">

  <!-- Uploaded image -->
  <img src=""
       id="uploadedPreview"
       class="preview-image">
</div>

</div>


  </div>

</form>
<script>
const fileInput = document.getElementById("fileInput");
const dummyGradcam = document.getElementById("dummyGradcam");
const uploadedPreview = document.getElementById("uploadedPreview");
const dropText = document.getElementById("dropText");
const predictBtn = document.querySelector(".predict-btn");

predictBtn.disabled = true;

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Show filename
  dropText.innerHTML = `<div class="upload-filename">${file.name}</div>`;

  // Load uploaded image
  const reader = new FileReader();
  reader.onload = e => {
    uploadedPreview.src = e.target.result;

    // SWITCH VISIBILITY
    dummyGradcam.classList.remove("show");
    uploadedPreview.classList.add("show");
  };
  reader.readAsDataURL(file);

  predictBtn.disabled = false;
});
</script>



</body>
</html>
"""


# ================= PREDICT =================
@app.post("/predict", response_class=HTMLResponse)
async def predict(image: UploadFile, organ: str = Form(...), gradcam: str = Form(None)):
    img = Image.open(image.file).convert("RGB").resize((224,224))
    x = transform(img).unsqueeze(0)
    x.requires_grad_(True)

    model = MODELS[organ]
    out = model(x)
    probs = torch.softmax(out,dim=1)
    conf, idx = torch.max(probs,dim=1)
    confidence = round(conf.item()*100,1)

    shape = "/static/liver_shape.png" if organ=="liver" else "/static/thyroid_shape.png"
    title = "Fatty Liver Prediction" if organ=="liver" else "Thyroid Prediction"

    if gradcam=="on":
        cam = GradCAM(model).generate(x,idx.item())
        heat = cv2.applyColorMap(np.uint8(255*cam),cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(np.array(img),0.6,heat,0.4,0)
        cv2.imwrite("static/gradcam.png",cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))

    rag = MEDICAL_INFO.get(organ,{}).get(LABELS[organ][idx.item()],{})
    show_cam = gradcam=="on"

    return f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
body{{
 margin:0;height:100vh;
 background:radial-gradient(circle at top,#111,#000);
 font-family:Inter;color:white;
 display:flex;align-items:center;justify-content:center;
}}
.wrapper{{
  width:1400px;
  display:grid;
  grid-template-columns:420px 360px 1fr;
  gap:56px;
}}

.card{{
 background:#0b0b0b;
 border-radius:24px;
 padding:32px;
   box-shadow:
    0 0 0 1px rgba(255,106,0,0.45),
    0 0 50px rgba(255,106,0,0.18);
}}
.shape-box{{
 width:260px;height:260px;
   background:#000;
  border:2px solid #ff6a00;
  box-shadow:0 0 0 1px rgba(255,255,255,0.15);
 mask:url('{shape}') center/contain no-repeat;
 -webkit-mask:url('{shape}') center/contain no-repeat;
 position:relative;
 overflow:hidden;
}}
.fill{{
 position:absolute;
 bottom:0;width:100%;
 height:{confidence}%;
 background:linear-gradient(180deg,#ff8a00,#ff6a00);
 animation:rise 2s ease;
  box-shadow:0 -10px 40px rgba(255,106,0,0.45);
}}
@keyframes rise{{from{{height:0}}}}
.percent{{
 position:absolute;inset:0;
 display:flex;align-items:center;justify-content:center;
 font-family:'Space Grotesk';
 font-size:32px;
}}
button{{
 padding:12px 22px;
 border-radius:999px;
 border:2px solid #ff6a00;
 background:transparent;
 color:#ff6a00;
 font-family:'Space Grotesk';
 cursor:pointer;
 margin-right:12px;
}}
button:hover{{background:#ff6a00;color:black}}
</style>
</head>

<body>
<div class="wrapper">

  <!-- LEFT: GRAD-CAM -->
  <div class="card">
    { "<h3>Grad-CAM</h3><img src='/static/gradcam.png'>" if show_cam else "" }
  </div>

  <!-- CENTER: ORGAN FILL -->
<div class="center">
  <h2>{title}</h2>

  <div class="shape-box">
    <div class="fill"></div>
    <div class="percent">{confidence}%</div>
  </div>

  <!-- ✅ ADD HERE -->
  <div style="margin-top:32px">
    <a href="/static/report.pdf">
      <button>Download</button>
    </a>

    <a href="/">
      <button>Predict another</button>
    </a>
  </div>
</div>


  <!-- RIGHT: EXPLANATION -->
  <div class="card">
    <h3>Medical Explanation</h3>
    <p>{rag.get("description","")}</p>
  </div>

</div>

</body>
</html>
"""
