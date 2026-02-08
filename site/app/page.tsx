import React from "react";

export default function Page() {
  return (
    <main className="page">
      <header>
        <h1 className="hero-title">Facial Recognition Neural Network</h1>
        <a href="https://github.com/nicholaschen09/facial-recognition-neural-network" target="_blank" rel="noopener noreferrer">GitHub</a>
      </header>

      <section className="section">
        <h2>The Big Picture</h2>
        <p>
          Modern devices unlock with your face in under a second. I wanted to
          rebuild a minimal version of that experience from scratch: collect a
          small dataset of faces, train a convolutional neural network, and use
          it to recognise whether the camera is currently seeing me
          (&quot;nic&quot;) or someone else (&quot;other&quot;).
        </p>
        <p>
          The project is organised as a small, production-style pipeline: raw
          images go through a preprocessing stage, get converted into clean
          face crops, are fed into a CNN for training, then evaluated and
          finally wired into a real-time webcam script.
        </p>
      </section>

      <section className="section">
        <h2>Pipeline Overview</h2>
        <h3>1. Raw Images → Processed Faces (`preprocess.py`)</h3>
        <p>
          The dataset starts as folders of images grouped by identity, for
          example:
        </p>
        <p>
          <code>dataset/train/nic</code> and <code>dataset/train/other</code>,
          plus matching folders for test data.
        </p>
        <p>
          <code>preprocess.py</code> walks these directories, runs OpenCV&apos;s{" "}
          <code>haarcascade_frontalface_default.xml</code> detector on each
          image, then:
        </p>
        <ul>
          <li>converts the frame to grayscale,</li>
          <li>crops a tight bounding box around each detected face,</li>
          <li>resizes to a fixed 96×96 resolution, and</li>
          <li>
            writes the crop into a mirrored{" "}
            <code>processed/train/&lt;class&gt;</code> structure.
          </li>
        </ul>
        <p>
          If no face is found, the script logs it so I can clean up low-quality
          or mislabeled images. By the end, every file in{" "}
          <code>processed/</code> is a clean, normalized input for the CNN.
        </p>

        <h3>2. CNN Model – `FaceRecognitionCNN` (`model.py`)</h3>
        <p>
          The core model is a compact convolutional neural network tailored for
          96×96 grayscale faces. The architecture is:
        </p>
        <ul>
          <li>
            Two convolutional blocks: <code>Conv → ReLU → MaxPool</code>,
            growing from 1 channel to 32, then 64 feature maps.
          </li>
          <li>
            A flattened feature vector of size <code>64 × 21 × 21</code> fed
            into a small fully connected head.
          </li>
          <li>
            A final linear layer outputting <code>num_classes</code> logits
            (here <code>[nic, other]</code>).
          </li>
        </ul>
        <p>
          Mathematically, the network learns a function that maps an input
          tensor of shape <code>1 × 96 × 96</code> to a 2‑dimensional score
          vector, where the argmax gives the predicted identity.
        </p>
      </section>

      <section className="section">
        <h2>Training & Evaluation</h2>
        <h3>3. Training Loop (`train.py`)</h3>
        <p>
          Training uses <code>torchvision.datasets.ImageFolder</code> on{" "}
          <code>../processed/train</code>, with transforms that:
        </p>
        <ul>
          <li>ensure images are grayscale,</li>
          <li>convert them to tensors, and</li>
          <li>normalize pixel values to a mean of 0.5 and std of 0.5.</li>
        </ul>
        <p>
          Hyperparameters are intentionally simple: batch size 32, 10 epochs,
          Adam optimizer with a learning rate of 1e‑3, and cross‑entropy loss.
          After each epoch the script prints the loss and, when training
          finishes, saves weights to <code>model.pt</code>.
        </p>

        <h3>4. Measuring Accuracy (`evaluate.py`)</h3>
        <p>
          For evaluation, I mirror the training setup but load{" "}
          <code>../processed/test</code> instead. The script restores{" "}
          <code>FaceRecognitionCNN</code> from <code>model.pt</code>, runs it
          on the test loader without gradient tracking, and reports:
        </p>
        <p>
          <code>Accuracy on test dataset: XX.XX%</code>
        </p>
        <p>
          This gives a clean, single metric for how well the system
          distinguishes Nic from everyone else on unseen data.
        </p>
      </section>

      <section className="section">
        <h2>Using the Model</h2>
        <h3>5. Single-Image Inference (`inference.py`)</h3>
        <p>
          To make the model easy to reuse, <code>inference.py</code> exposes a
          small <code>predict(image_path)</code> helper. It:
        </p>
        <ul>
          <li>loads the trained CNN with <code>num_classes = 2</code>,</li>
          <li>applies the same preprocessing transforms as training, and</li>
          <li>
            returns the human‑readable label from{" "}
            <code>[&quot;nic&quot;, &quot;other&quot;]</code>.
          </li>
        </ul>

        <h3>6. Real-Time Webcam Recognition (`webcam.py`)</h3>
        <p>
          The most satisfying part is the webcam demo. It uses OpenCV to grab
          frames from <code>VideoCapture(0)</code>, runs the same Haar Cascade
          detector, feeds each detected face through the CNN, and overlays:
        </p>
        <ul>
          <li>a bounding box around the face, and</li>
          <li>a label saying either &quot;nic&quot; or &quot;other&quot;.</li>
        </ul>
        <p>
          Hit <code>q</code> to exit, and you&apos;ve effectively turned your
          laptop into a tiny, on-device facial recognition system.
        </p>
      </section>

      <section className="section">
        <h2>How to Run It Yourself</h2>
        <p>
          From the Python project root, you can reproduce the full pipeline:
        </p>
        <ul>
          <li>
            Preprocess faces: <code>python preprocess.py</code>
          </li>
          <li>
            Train: <code>python train.py</code>
          </li>
          <li>
            Evaluate: <code>python evaluate.py</code>
          </li>
          <li>
            Webcam demo: <code>python webcam.py</code>
          </li>
        </ul>
        <p>
          This Next.js page is just the front-end storytelling layer — the real
          work happens in the PyTorch code under the hood.
        </p>
      </section>

      <section className="section">
        <h2>Takeaways</h2>
        <p>
          Building this project made it clear how much impact careful
          preprocessing and consistent transforms have on model quality. Even a
          relatively small CNN can perform surprisingly well when every face is
          aligned, normalized, and seen through the same lens during training
          and inference. More than anything, wiring the model into a live
          webcam loop made the whole thing feel real — turning abstract tensors
          and loss curves into an interactive tool that either recognises me or
          confidently says &quot;other&quot;.
        </p>
      </section>

      <footer className="footer">
        <div>
          Built by Nicholas Chen · <a href="https://github.com/nicholaschen09" target="_blank" rel="noopener noreferrer">GitHub</a>
        </div>
      </footer>
    </main>
  );
}


