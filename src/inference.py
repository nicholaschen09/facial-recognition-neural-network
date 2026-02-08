import torch
from torchvision import transforms
from PIL import Image
from model import FaceRecognitionCNN

# Load the model
num_classes = 2
model = FaceRecognitionCNN(num_classes)
model.load_state_dict(torch.load('model.pt'))
model.eval()

class_names = ['nic', 'other']

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
    return class_names[predicted.item()]

