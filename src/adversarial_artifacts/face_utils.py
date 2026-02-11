import cv2
import numpy as np
import torch
import requests
from torchvision import models, transforms
from collections import deque 
from pathlib import Path
from . import config

class AdversarialProcessor:
    """
    Handles face detection, patch overlay, and AI model inference.
    """
    def __init__(self):
        # --- Path and State Setup ---
        self.mode = config.INITIAL_MODE 
        self.show_patch = True
        self.smooth_box = deque(maxlen=config.FACE_SMOOTHING_BUFFER)

        # --- Load Data and Models ---
        self._load_haarcascade()
        self.patches = self._load_patches()
        self.imagenet_labels = self._load_imagenet_labels()
        self.model, self.device, self.normalize = self._load_ai_model()

    def _load_haarcascade(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + config.HAARCASCADE_FILE)

    def _load_patches(self):
        """Loads the two specific patch files from the data directory."""
        
        def load_patch_from_file(filename):
            if not filename: return None
            try:
                # Use importlib.resources for robust path handling within packages
                from importlib import resources
                with resources.files('adversarial_artifacts').joinpath('data', 'patches', filename) as path:
                    if not path.exists():
                        print(f"Warning: Patch file not found at {path}")
                        return None
                    img = cv2.imread(str(path))
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
            except (ImportError, ModuleNotFoundError, AttributeError): # Fallback for older python
                project_root = Path(__file__).parent.parent.parent
                path = project_root / "data" / "patches" / filename
                img = cv2.imread(str(path))
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
            except Exception as e:
                print(f"Error loading patch {filename}: {e}")
                return None

        patches = {
            "TARGETED": load_patch_from_file(config.PATCH_FILES.get("TARGETED")),
            "UNTARGETED": load_patch_from_file(config.PATCH_FILES.get("UNTARGETED"))
        }

        if patches["UNTARGETED"] is None and patches["TARGETED"] is None:
            raise FileNotFoundError("Could not load any patch files. Please check 'src/adversarial_artifacts/data/patches'.")
        return patches

    def _load_imagenet_labels(self):
        try:
            return requests.get(config.AI_IMAGENET_LABELS_URL).text.splitlines()
        except requests.exceptions.RequestException as e:
            print(f"Error downloading ImageNet labels: {e}. AI labels will be unavailable.")
            return []

    def _load_ai_model(self):
        device_name = config.AI_DEVICE if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        print(f"Using device: {device}")
        model = models.resnet50(weights='DEFAULT').to(device).eval()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return model, device, normalize

    def switch_mode(self):
        self.mode = "TARGETED" if self.mode == "UNTARGETED" else "UNTARGETED"
        print(f"Switched to {self.mode} mode.")

    def toggle_patch_visibility(self):
        self.show_patch = not self.show_patch
        print(f"Patch visibility: {'ON' if self.show_patch else 'OFF'}")

    def process_frame(self, frame):
        """Detects faces, applies overlays, and runs AI inference."""
        display_frame = frame.copy()
        h_frame, w_frame, _ = frame.shape
        ui_data = {}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            config.FACE_DETECT_SCALE_FACTOR,
            config.FACE_DETECT_MIN_NEIGHBORS
        )

        if len(faces) > 0:
            (x_raw, y_raw, w_raw, h_raw) = max(faces, key=lambda b: b[2] * b[3])
            self.smooth_box.append((x_raw, y_raw, w_raw, h_raw))
            x, y, w, h_face = np.mean(self.smooth_box, axis=0).astype(int)

            # --- Overlay Logic (from notebook) ---
            current_patch = self.patches.get(self.mode)
            if self.show_patch and current_patch is not None:
                bril_w = w
                bril_h = int(bril_w * (current_patch.shape[0] / current_patch.shape[1]))
                y_offset = y + int(h_face * config.PATCH_Y_OFFSET_FACTOR)
                
                if 0 <= y_offset and y_offset + bril_h < h_frame and 0 <= x and x + bril_w < w_frame:
                    overlay_resized = cv2.resize(current_patch, (bril_w, bril_h))
                    overlay_mask = np.sum(overlay_resized, axis=2) < config.PATCH_MASK_THRESHOLD
                    roi = display_frame[y_offset:y_offset+bril_h, x:x+bril_w]
                    if roi.shape[:2] == overlay_resized.shape[:2]:
                        roi[overlay_mask] = cv2.cvtColor(overlay_resized, cv2.COLOR_RGB2BGR)[overlay_mask]

            # --- AI Logic (from notebook) ---
            analysis_source = display_frame if self.show_patch else frame
            face_zone = analysis_source[y:y+h_face, x:x+w]
            
            if face_zone.size > 0:
                face_resized = cv2.resize(cv2.cvtColor(face_zone, cv2.COLOR_BGR2RGB), (224, 224))
                input_tensor = self.normalize(torch.from_numpy(face_resized).permute(2, 0, 1).float().div(255).to(self.device)).unsqueeze(0)

                with torch.no_grad():
                    output = self.model(input_tensor)
                    probs = torch.softmax(output[0], dim=0)

                    result_data = {}
                    if "TARGETED" in self.mode:
                        p_prob = probs[config.AI_TARGET_ID_PANDA].item()
                        h_prob = torch.max(probs[torch.tensor(config.AI_HUMAN_IDS)]).item()
                        if p_prob > h_prob:
                            result_data = {'label': 'PANDA', 'prob': p_prob, 'success': True}
                        else:
                            result_data = {'label': 'HUMAN', 'prob': h_prob, 'success': False}
                    else: # UNTARGETED
                        if self.imagenet_labels:
                            animal_probs = probs[config.AI_ANIMAL_IDS]
                            best_prob = torch.max(animal_probs).item()
                            best_id = config.AI_ANIMAL_IDS[torch.argmax(animal_probs).item()]
                            result_data = {'label': self.imagenet_labels[best_id], 'prob': best_prob}
                        else:
                            result_data = {'label': 'Labels not loaded', 'prob': 0}
                
                if result_data:
                    ui_data['classification'] = {'result': result_data, 'pos': (x, y - 15)}

        return display_frame, ui_data