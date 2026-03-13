import os
import shutil
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------

MODEL_PATH = "face_quality_classifier.pt"
IMAGE_DIR = "./test_dataset_preprocessed/1024"
OUTPUT_DIR = "./test_dataset_preprocessed/filter"

IMAGE_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# confidence bins
THRESHOLDS = [0.90]

# ImageNet normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# -----------------------------
# Preprocess
# -----------------------------

def preprocess(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)

    arr = np.asarray(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1)

    tensor = (tensor - MEAN) / STD
    return tensor


# -----------------------------
# Classifier definition
# -----------------------------

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()


# -----------------------------
# Load models
# -----------------------------

print("Loading DINOv2 backbone...")
backbone = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitb14"
).to(DEVICE)

backbone.eval()

print("Loading classifier...")
classifier = Classifier().to(DEVICE)
classifier.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
classifier.eval()

# -----------------------------
# Prepare folders
# -----------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_folder(conf):
    for t in THRESHOLDS:
        if conf >= t:
            return f"confidence_{int(t*100)}"
    return "confidence_0"


for t in THRESHOLDS:
    os.makedirs(os.path.join(OUTPUT_DIR, f"confidence_{int(t*100)}"), exist_ok=True)

os.makedirs(os.path.join(OUTPUT_DIR, "confidence_0"), exist_ok=True)

# -----------------------------
# Inference loop
# -----------------------------

files = sorted(os.listdir(IMAGE_DIR))

with torch.no_grad():
    for name in tqdm(files):
        path = os.path.join(IMAGE_DIR, name)
        img = preprocess(path).unsqueeze(0).to(DEVICE)

        # DINOv2 embedding
        embedding = backbone(img)

        # classifier prediction
        logit = classifier(embedding)
        confidence = torch.sigmoid(logit).item()
        folder = get_folder(confidence)
        dst = os.path.join(OUTPUT_DIR, folder, name)
        shutil.copy2(path, dst)


print("Done.")