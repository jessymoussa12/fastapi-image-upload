from PIL import Image
from io import BytesIO
from fastapi import HTTPException

import torch
from torchvision import models, transforms


# Load the pre-trained ResNet18 model when the application starts.
# This avoids loading the model every time a prediction is requested.
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Put the model into evaluation mode because we are only making predictions,
# not training the neural network.
model.eval()


# Define the image preprocessing pipeline.
# ResNet18 expects images to be resized and converted into tensors.
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Load the ImageNet category labels.
# These labels allow us to convert prediction indexes into readable names.
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

    # First validate the uploaded image.
    validate_image(contents)

    # Open the image and convert it to RGB.
    # This ensures a consistent 3-channel format for the model.
    image = Image.open(BytesIO(contents)).convert("RGB")

    # Apply preprocessing:
    # resize the image and convert it into a tensor.
    image_tensor = preprocess(image)

    # Add a batch dimension because ResNet18 expects a batch of images,
    # even if we are only sending one image.
    image_tensor = image_tensor.unsqueeze(0)

    # Run inference with the model.
    # no_grad() improves performance by disabling gradient calculations.
    with torch.no_grad():
        outputs = model(image_tensor)

    # Convert the model's raw scores into probabilities.
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Find the category with the highest probability.
    predicted_index = torch.argmax(probabilities).item()

    # Convert the category index into a human-readable label.
    prediction = class_names[predicted_index]

    # Extract the confidence score for the predicted category.
    confidence = float(probabilities[predicted_index])

    # Return the final prediction and confidence score.
    return {
        "prediction": prediction,
        "confidence": confidence
    }
