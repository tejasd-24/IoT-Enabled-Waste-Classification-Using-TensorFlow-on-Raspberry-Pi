#!/usr/bin/env python3


import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


IMAGE_PATH = "/home/rtlguy/Downloads/bio.jpg"
MODEL_PATH = "/home/rtlguy/Desktop/waste_classifier_model.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WasteClassifier(nn.Module):
    def __init__(self, num_classes=2, backbone='resnet18', pretrained=True):
        super(WasteClassifier, self).__init__()
        if backbone == 'resnet18':
            self.backbone = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=None if not pretrained else models.ResNet50_Weights.DEFAULT)
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        elif backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(weights=None if not pretrained else models.EfficientNet_B0_Weights.DEFAULT)
            self.backbone.classifier[1] = nn.Linear(self.backbone.classifier[1].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)


transform = transforms.Compose([
    transforms.Resize((200, 200)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


image = Image.open(IMAGE_PATH).convert('RGB')
image = transform(image).unsqueeze(0)  # Add batch dimension
image = image.to(DEVICE)


model = WasteClassifier(num_classes=2, backbone='resnet18', pretrained=False)
model = model.to(DEVICE)
model.eval()

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)


if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)


with torch.no_grad():
    outputs = model(image)
    probs = F.softmax(outputs, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
    prob_values = probs.squeeze().cpu().numpy()

classes = ['Non-biodegradable', 'Biodegradable']
print(f"Predicted Class: {classes[pred_class]}")
print(f"Probabilities: Non-bio={prob_values[0]:.4f}, Bio={prob_values[1]:.4f}")

