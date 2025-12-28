from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import numpy as np

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import NeuralNetwork
NN = NeuralNetwork(load_path="../../checkpoints/Exp3/model_parameters_86.npz")

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

@api.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess
    image = image.convert("RGB")
    image = image.resize((64, 64), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0 # (64, 64, 3) RGB
    arr = arr[:, :, ::-1] # Network learned in BGR
    arr = arr.transpose(2, 0, 1) # (H, W, C) -> (C, H, W)
    arr = arr.reshape(-1, 1)

    # Run inference
    _, probabilities = NN.predict(arr)

    return JSONResponse({
        "dog": probabilities[0, 0],
        "dough": probabilities[1, 0]
    })