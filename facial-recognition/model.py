import torch
import torch.nn as nn

class FaceRecognitionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2),  # Output: (32, 46, 46)
            nn.Conv2d(32, 64, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2)  # Output: (64, 21, 21)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 21 * 21, 256), nn.ReLU(),  # Adjusted input size
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        print(f"Shape after convolutional layers: {x.shape}")  # Debugging
        x = self.classifier(x)
        return x