# Adversarial Artifacts Demo

This project is a real-time, interactive demonstration of adversarial attacks against a deep learning model for image classification. Using a webcam, it detects a user's face and overlays a specially crafted "adversarial patch" (designed to look like glasses) to fool a pre-trained ResNet50 model.

## How It Works

The application captures video from a webcam, uses a Haar Cascade classifier to find faces in real-time, and then performs two main tasks:

1.  **Overlay:** It places an adversarial patch over the detected face.
2.  **Inference:** It runs the ResNet50 model on the face region (with or without the patch) and displays the model's top classification prediction.

The goal is to visually demonstrate how a physical object can manipulate a computer vision system's output.

### Attack Modes

The demo operates in two distinct modes:

- **TARGETED Attack:** The patch is designed to make the model misclassify a human face specifically as a **"giant panda"**. The UI will indicate success (green text) if the panda classification is more probable than any human-related class.
- **UNTARGETED Attack:** The patch is designed to cause a misclassification to _any_ class that is not the correct one. In this demo, it attempts to classify the face as one of several pre-selected animal classes.

## Features

- Real-time face detection and classification.
- Switch between **TARGETED** and **UNTARGETED** attack modes.
- Toggle the visibility of the adversarial patch to see the difference in classification.
- Record the live demo output to an MP4 file.
- Smooth video processing using a threaded camera implementation for better performance.

## Installation

To run this demo, you'll need Python 3.7+ and Git. It is highly recommended to use a virtual environment.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/adversarial-artifacts-demo.git
    cd adversarial-artifacts-demo
    ```

2.  **Create and activate a virtual environment:**
    - On Windows:
      ```bash
      python -m venv venv
      .\venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      python -m venv venv
      source venv/bin/activate
      ```

3.  **Install the package and its dependencies:**
    The project is set up as a Python package. Install it in editable mode (`-e`) to ensure the application can find all its internal resources correctly.
    ```bash
    pip install -e .
    ```
    This command reads the `setup.py` file and installs all required libraries, such as PyTorch, OpenCV, and NumPy.

## Usage

Once the package is installed in your virtual environment, you can run the demo directly from your terminal using the console script entry point:

```bash
run_demo
```

The application will start in fullscreen mode, using the default camera.

### Keyboard Controls

Use the following keys while the demo is running:

| Key        | Action                                            |
| :--------- | :------------------------------------------------ |
| `q`        | Quit the application.                             |
| `spacebar` | Toggle the adversarial patch on/off.              |
| `t`        | Switch between `TARGETED` and `UNTARGETED` modes. |
| `r`        | Start or stop recording the video output.         |

Recordings are saved in the `recordings/` directory, which is created automatically in your project folder.

## Project Structure

```
adversarial-artifacts-demo/
├── .gitignore
├── README.md
├── setup.py
├── src/
│   └── adversarial_artifacts/
│       ├── __init__.py
│       ├── camera_utils.py       # Threaded camera management
│       ├── config.py             # All configuration constants
│       ├── demo.py               # Main application loop and UI rendering
│       ├── face_utils.py         # Core logic for detection, overlay, and inference
│       └── data/
│           └── patches/
│               ├── targeted_bril_PIXEL_PERFECT.png
│               └── untargeted_bril_PIXEL_PERFECT.png
└── recordings/                     # Output directory for videos (created on-the-fly)
```
