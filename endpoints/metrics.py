import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from torchvision import transforms
from metrics.calculator import MetricsCalculator
from PIL import Image
import base64
import io


class ImagePair(BaseModel):
    original: str  
    reconstructed: str  

def decode_base64_image(data: str) -> torch.Tensor:
    image_data = base64.b64decode(data)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = image.resize((425, 425), Image.LANCZOS)

    transform = transforms.ToTensor()
    return transform(image).unsqueeze(0)

router = APIRouter(prefix="/metrics", tags=["Metrics"])

@router.post("/")
def compute_metrics(pair: ImagePair, calculator : MetricsCalculator):
    img1 = decode_base64_image(pair.original).unsqueeze(0)
    img2 = decode_base64_image(pair.reconstructed).unsqueeze(0)

    metrics = calculator.compute_all(img1, img2)
    return metrics