
import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import transformers.generation
from torchvision import datasets, transforms, models
from google.colab import drive
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
import torchvision.transforms as T
from PIL import Image
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
import asyncio
import nest_asyncio
import timm
from transformers import pipeline

!pip install nest_asyncio
!pip install aiogram
!pip install bitsandbytes
!pip install transformers_stream_generator
!pip install dependencies
!pip install dependencies
!pip uninstall torch torchvision torchaudio timm -y
!pip cache purge
!pip install torch torchvision torchaudio timm

import torch


drive.mount('/content/drive')


ZIP_PATH = "/content/drive/MyDrive/skin.zip"
EXTRACT_DIR = "skin_data" 
OUT_DIR = "checkpoints" 


if not os.path.exists(EXTRACT_DIR):
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    print(f"Архив распакован в директорию: {EXTRACT_DIR}")
else:
    print(f"Директория {EXTRACT_DIR} уже существует")

DATA_DIR = os.path.join(EXTRACT_DIR, "train") 
os.makedirs("checkpoints", exist_ok=True)

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

DATA_DIR = "skin_data"   
SAVE_DIR = "trained_models"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = ["akne", "eksim", "herpes", "panu", "rosacea"]
NUM_CLASSES = len(CLASS_NAMES)

train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


train_ds = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
val_size = int(0.2 * len(train_ds))
train_size = len(train_ds) - val_size
train_ds, val_ds = torch.utils.data.random_split(train_ds, [train_size, val_size])

val_ds.dataset.transform = val_tfms  

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

def get_model():
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=NUM_CLASSES)
    return model.to(DEVICE)


def train_one_model(model, epochs=EPOCHS):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

  
        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(true_labels, preds)
        print(f"[EfficientNet-B3] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")

    
        history["train_loss"].append(avg_loss)
        history["val_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(SAVE_DIR, "efficientnet_b3.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Сохранена лучшая модель EfficientNet-B3 с точностью {best_acc:.4f}")

    return history

if __name__ == "__main__":
    model = get_model()
    history = train_one_model(model)
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss", marker="o")
    plt.plot(history["val_acc"], label="Validation Accuracy", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Значение")
    plt.title("График обучения EfficientNet-B3")
    plt.legend()
    plt.grid(True)
    plt.show()

model_name = "Qwen/Qwen-7B"


tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
qwen_pipe = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",       
    trust_remote_code=True,
    load_in_4bit=True         
)


TELEGRAM_TOKEN = "ТОКЕН"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_NAMES = ["akne", "eksim", "herpes", "panu", "rosacea"]
NUM_CLASSES = len(CLASS_NAMES)


def get_model(name, weight_path=None):
    model = timm.create_model(name, pretrained=False, num_classes=NUM_CLASSES)
    if weight_path and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        print(f"✅ Загружены веса для {name} из {weight_path}")
    return model.to(DEVICE).eval()

skin_model = get_model("efficientnet_b3", "trained_models/efficientnet_b3.pth")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
])


def predict_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = skin_model(img)
            _, pred_idx = torch.max(outputs, 1)

        return CLASS_NAMES[pred_idx.item()]
    except Exception as e:
        print("Ошибка классификации:", e)
        return "Ошибка"

def get_recommendation(disease: str) -> str:
    try:
        prompt = f" {disease}"
        result = qwen_pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.1)
        return result[0]["generated_text"].replace(prompt, "Диагноз: akne\nВопрос: Какие советы по питанию и уходу за кожей помогут при акне? Напиши список кратко, без дозировок, укажи источники (Кратко).").strip()
    except Exception as e:
        return f"Ошибка генерации: {e}"

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer("Привет! Отправь фото кожи, и я попробую определить заболевание и дать рекомендации.")

@dp.message()
async def handle_message(message: types.Message):
    if not message.photo:
        await message.answer("Пожалуйста, отправь фото кожи.")
        return

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    downloaded_file = await bot.download_file(file_path)

    img_path = "temp.jpg"
    with open(img_path, "wb") as f:
        f.write(downloaded_file.read())

    diagnosis = predict_image(img_path)

    if diagnosis == "Ошибка":
        await message.answer("Ошибка: не удалось обработать фото.")
    elif diagnosis not in CLASS_NAMES:
        await message.answer("Нету")
    else:
        recs = get_recommendation(diagnosis)
        await message.answer(f"Определено заболевание: {diagnosis}\n\n💊 Рекомендации:\n{recs}")

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    nest_asyncio.apply()
    asyncio.get_event_loop().run_until_complete(main())
