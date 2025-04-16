import torch
from torchvision import transforms
from PIL import Image
from model import FaceRecognitionCNN

# Load model
model = FaceRecognitionCNN(num_classes=10)
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Preprocess image
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Example usage:
print(predict('processed/test/person1/img1.jpg'))