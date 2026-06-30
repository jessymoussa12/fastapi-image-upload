from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

model.eval()


def predict_image():
    return {
        "prediction": "cat",
        "confidence": 0.95
    }
