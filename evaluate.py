import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import FaceRecognitionCNN

# Hyperparameters
batch_size = 32

# Data loaders
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_dataset = datasets.ImageFolder('../processed/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the model
num_classes = len(test_dataset.classes)
model = FaceRecognitionCNN(num_classes)
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test dataset: {accuracy:.2f}%")