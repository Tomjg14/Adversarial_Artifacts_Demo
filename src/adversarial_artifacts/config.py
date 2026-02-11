import cv2

# --- General Settings ---
INITIAL_MODE = "UNTARGETED"  # "TARGETED" or "UNTARGETED"

# --- Camera Settings ---
CAMERA_INDEX = 0
CAMERA_WARMUP_TIME = 1.0  # seconds

# --- Data Paths ---
HAARCASCADE_FILE = "haarcascade_frontalface_default.xml"
PATCH_FILES = {
    "TARGETED": "targeted_bril_PIXEL_PERFECT.png",
    "UNTARGETED": "untargeted_bril_PIXEL_PERFECT.png"
}

# --- Face Detection & Processing ---
FACE_SMOOTHING_BUFFER = 7
FACE_DETECT_SCALE_FACTOR = 1.3
FACE_DETECT_MIN_NEIGHBORS = 5

# --- Patch Overlay Settings ---
PATCH_Y_OFFSET_FACTOR = 0.20  # Percentage of face height to offset patch
PATCH_MASK_THRESHOLD = 700     # For non-alpha images, color value sum to be considered transparent

# --- AI Model Settings ---
AI_DEVICE = "cuda"  # "cuda" or "cpu"
AI_IMAGENET_LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
# Class IDs for UNTARGETED mode
AI_ANIMAL_IDS = [388, 281, 282, 285, 151, 153, 254, 291, 340, 386, 385]
# Class IDs for TARGETED mode
AI_TARGET_ID_PANDA = 388
AI_HUMAN_IDS = [834, 835, 652] # IDs for 'person', 'groom', 'bride' etc. to compare against

# --- UI Settings ---
WINDOW_NAME = 'Demo'
# Font settings
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_MODE_TEXT_POS = (20, 50)
UI_MODE_TEXT_SCALE = 1.0
UI_MODE_TEXT_THICKNESS = 2
UI_MODE_TEXT_OUTLINE_THICKNESS = 3
UI_CLASSIFICATION_SCALE = 1.2
UI_CLASSIFICATION_THICKNESS = 3
UI_COLOR_WHITE = (255, 255, 255)
UI_COLOR_BLACK = (0, 0, 0)
UI_COLOR_RED = (0, 0, 255)
UI_COLOR_GREEN = (0, 255, 0)
UI_COLOR_CYAN = (255, 255, 0)

# --- Recording Settings ---
RECORDING_DIR = "recordings"
RECORDING_FILENAME_PREFIX = "demo_output"
RECORDING_FPS = 20.0
RECORDING_FOURCC = 'mp4v'
REC_INDICATOR_POS_OFFSET = (50, 50)
REC_INDICATOR_RADIUS = 20
REC_INDICATOR_TEXT_OFFSET = (130, 60)
REC_INDICATOR_TEXT = "REC"
REC_INDICATOR_TEXT_SCALE = 1.0
REC_INDICATOR_TEXT_THICKNESS = 2