# Facial Recognition Neural Network

A facial recognition system built with PyTorch. It preprocesses images, trains a CNN, evaluates accuracy, and runs real-time recognition via webcam.

## Project Structure

```
├── src/                  # Python ML pipeline
│   ├── model.py          # CNN architecture (FaceRecognitionCNN)
│   ├── preprocess.py     # Face detection & cropping
│   ├── train.py          # Training loop
│   ├── evaluate.py       # Test-set accuracy
│   ├── inference.py      # Single-image prediction
│   └── webcam.py         # Real-time webcam demo
├── site/                 # Next.js frontend (project write-up)
│   ├── app/
│   │   ├── page.tsx
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── package.json
│   └── ...
├── requirements.txt      # Python dependencies
├── vercel.json           # Vercel deployment config
└── README.md
```

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd facial-recognition-neural-network
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare your dataset

Place images in the following structure:

```
dataset/
├── train/
│   ├── nic/
│   └── other/
└── test/
    ├── nic/
    └── other/
```

## Usage

All Python scripts should be run from the `src/` directory:

```bash
cd src
```

**Preprocess faces** — detect, crop, and resize to 96x96 grayscale:

```bash
python preprocess.py
```

**Train the model:**

```bash
python train.py
```

**Evaluate accuracy on the test set:**

```bash
python evaluate.py
```

**Run the real-time webcam demo:**

```bash
python webcam.py
```

Press `q` to quit the webcam window.

## Site

The `site/` directory contains a Next.js app that serves as a project write-up. To run it locally:

```bash
cd site
npm install
npm run dev
```
