import cv2
import torch
from model import FaceRecognitionCNN
from torchvision import transforms
from PIL import Image

model = FaceRecognitionCNN(num_classes=2)
model.load_state_dict(torch.load('model.pt'))
model.eval()

class_names = ['nic', 'other'] 

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (96, 96))
        face_pil = Image.fromarray(face_resized)
        face_tensor = transform(face_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(face_tensor)
            _, predicted = torch.max(output, 1)
        label = class_names[predicted.item()] 
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()