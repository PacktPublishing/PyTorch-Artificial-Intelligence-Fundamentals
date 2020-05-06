import io
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms
import json

with open('idx_class.json') as f:
    idx_class = json.load(f)


def create_model():

    model_path = "densenet161.pth"
    model = models.densenet161(pretrained=True)
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.eval()
    return model

def image_transformer(image_data):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_data))
    return transform(image).unsqueeze(0)

def predict_image(model, image_data):

    image_tensor = image_transformer(image_data)
    output = model(image_tensor)
    _, prediction = output.max(1)
    object_index = prediction.item()

    return idx_class[object_index]
