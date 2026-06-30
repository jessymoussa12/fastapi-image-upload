from PIL import Image
from io import BytesIO
from fastapi import HTTPException

import torch
from torchvision import models, transforms

# Load the pre-trained model once when the application starts
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.eval()

# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ImageNet class names
class_names = models.ResNet18_Weights.DEFAULT.meta["categories"]


def validate_image(contents: bytes):
    try:
        image = Image.open(BytesIO(contents))
        image.verify()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be opened as a valid image."
        )


def predict_image_class(contents: bytes):
    validate_image(contents)

    image = Image.open(BytesIO(contents)).convert("RGB")

    image_tensor = preprocess(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    predicted_index = torch.argmax(probabilities).item()

    prediction = class_names[predicted_index]

    confidence = float(probabilities[predicted_index])

    return {
        "prediction": prediction,
        "confidence": confidence
    }
