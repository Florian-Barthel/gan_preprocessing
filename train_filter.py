import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


IMAGE_DIR = "/nvme1/face_datasets/FFHQ/FFHQC/256/"
GOOD_JSON = "/nvme1/face_datasets/FFHQ/FFHQC/custom_dist/smile_pose_rebalancing_v4_100k.json"
BAD_JSON = "./multi_face_index_list.json"
EMBEDDING_FILE = "embeddings.pt"

BATCH_SIZE = 256
EPOCHS = 10
LR = 1e-3
VAL_SPLIT = 0.01
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization (required for DINOv2)
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# -----------------------------
# Convert JSON ids → filenames
# -----------------------------

with open(GOOD_JSON) as f:
    good_ids = list(set(json.load(f)))

with open(BAD_JSON) as f:
    bad_ids = list(set(json.load(f)))

good_images = set(f"{i:05d}.jpg" for i in good_ids if i not in bad_ids)

# -----------------------------
# Image preprocessing
# -----------------------------

def preprocess_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2,0,1)
    tensor = (tensor - MEAN) / STD

    return tensor

# -----------------------------
# Dataset
# -----------------------------

class ImageDataset(Dataset):
    def __init__(self, image_dir, good_images):
        self.image_dir = image_dir
        self.files = sorted(os.listdir(image_dir))
        self.labels = [1 if f in good_images else 0 for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        path = os.path.join(self.image_dir, name)
        img = preprocess_image(path)
        label = self.labels[idx]
        return img, label, name


dataset = ImageDataset(IMAGE_DIR, good_images)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Step 1: Compute embeddings
# -----------------------------

if not os.path.exists(EMBEDDING_FILE):
    print("Computing DINOv2 embeddings...")
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(DEVICE)
    backbone.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in tqdm(loader):
            imgs = imgs.to(DEVICE)
            feats = backbone(imgs)
            all_embeddings.append(feats.cpu())
            all_labels.extend(labels)

    embeddings = torch.cat(all_embeddings)
    labels = torch.tensor(all_labels).float()

    torch.save({"embeddings": embeddings, "labels": labels}, EMBEDDING_FILE)
    print("Embeddings saved.")


data = torch.load(EMBEDDING_FILE)
embeddings = data["embeddings"]
labels = data["labels"]


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


embed_dataset = EmbeddingDataset(embeddings, labels)

# split
val_size = int(len(embed_dataset) * VAL_SPLIT)
train_size = len(embed_dataset) - val_size

train_ds, val_ds = random_split(embed_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)


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


model = Classifier().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def evaluate():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            preds = torch.sigmoid(logits) > 0.5
            correct += (preds == y.bool()).sum().item()
            total += y.size(0)
    return correct / total


best_acc = 0
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    acc = evaluate()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "face_quality_classifier.pt")
    print(f"Epoch {epoch+1}/{EPOCHS} | val acc {acc:.4f}")

print("Training complete.")
