import cv2
import os

def preprocess_images(input_dir, output_dir, img_size=(96, 96)):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for root, _, files in os.walk(input_dir):
        for file in files:
            img_path = os.path.join(root, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                print(f"No faces detected in: {img_path}")
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, img_size)
                output_path = os.path.join(output_dir, os.path.relpath(img_path, input_dir))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, face_resized)
                print(f"Successfully processed and saved: {output_path}")

# Process all subdirectories in train and test
for dataset_type in ['train', 'test']:
    input_base_dir = f'../dataset/{dataset_type}' 
    output_base_dir = f'../processed/{dataset_type}'
    for subdir in os.listdir(input_base_dir):
        input_dir = os.path.join(input_base_dir, subdir)
        output_dir = os.path.join(output_base_dir, subdir)
        if os.path.isdir(input_dir): 
            print(f"Processing directory: {input_dir}") 
            preprocess_images(input_dir, output_dir)