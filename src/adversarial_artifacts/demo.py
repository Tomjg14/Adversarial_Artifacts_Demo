import cv2
import time
from pathlib import Path
from adversarial_artifacts.camera_utils import ThreadedCamera
from adversarial_artifacts.face_utils import AdversarialProcessor
from . import config

def _setup_application(camera_index):
    """Initializes camera, processor, and window settings."""
    cam = ThreadedCamera(camera_index).start()
    time.sleep(config.CAMERA_WARMUP_TIME)  # Allow camera to initialize
    if not cam or not cam.grabbed:
        print("Failed to start camera. Please check camera index and permissions.")
        return None, None, None, None

    processor = AdversarialProcessor()
    w_frame = int(cam.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cam.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    if config.UI_FULLSCREEN:
        cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    return cam, processor, (w_frame, h_frame), None

def _draw_ui(frame, processor, ui_data, is_recording, frame_dims):
    """Draws all UI elements onto the frame."""
    w_frame, _ = frame_dims
    mode_text = f"Mode: {processor.mode}"
    
    # Draw mode text with a black outline for visibility
    cv2.putText(frame, mode_text, config.UI_MODE_TEXT_POS, config.UI_FONT, 
                config.UI_MODE_TEXT_SCALE, config.UI_COLOR_BLACK, config.UI_MODE_TEXT_OUTLINE_THICKNESS)
    cv2.putText(frame, mode_text, config.UI_MODE_TEXT_POS, config.UI_FONT, 
                config.UI_MODE_TEXT_SCALE, config.UI_COLOR_WHITE, config.UI_MODE_TEXT_THICKNESS)

    if 'classification' in ui_data:
        info = ui_data['classification']
        result = info['result']
        pos = info['pos']
        
        label_text = f"{result.get('label', 'N/A')} ({result.get('prob', 0)*100:.1f}%)"
        
        # Determine color based on mode and success
        color = config.UI_COLOR_BLACK # default
        if processor.mode == "TARGETED":
            if processor.show_patch:
                # Attack is active: Green for success, Red for failure.
                color = config.UI_COLOR_GREEN if result.get('success') else config.UI_COLOR_RED
            else:
                # Attack is inactive: Use a neutral color for natural classification.
                color = config.UI_COLOR_WHITE
        else: # UNTARGETED
            color = config.UI_COLOR_CYAN
            
        cv2.putText(frame, label_text, pos, config.UI_FONT, 
                    config.UI_CLASSIFICATION_SCALE, color, config.UI_CLASSIFICATION_THICKNESS)

    if is_recording:
        rec_pos = (w_frame - config.REC_INDICATOR_POS_OFFSET[0], config.REC_INDICATOR_POS_OFFSET[1])
        cv2.circle(frame, rec_pos, config.REC_INDICATOR_RADIUS, config.UI_COLOR_RED, -1)
        
        rec_text_pos = (w_frame - config.REC_INDICATOR_TEXT_OFFSET[0], config.REC_INDICATOR_TEXT_OFFSET[1])
        cv2.putText(frame, config.REC_INDICATOR_TEXT, rec_text_pos, config.UI_FONT, 
                    config.REC_INDICATOR_TEXT_SCALE, config.UI_COLOR_RED, config.REC_INDICATOR_TEXT_THICKNESS)

def _handle_key_press(key, processor, is_recording, video_writer, frame_dims, is_fullscreen):
    """Handles user key presses and returns application state."""
    should_quit = False
    if key == ord('q'):
        should_quit = True
    elif key == ord(' '):  # Space bar
        processor.toggle_patch_visibility()
    elif key == ord('t'):
        processor.switch_mode()
    elif key == ord('r'):
        if not is_recording:
            recordings_dir = Path(config.RECORDING_DIR)
            recordings_dir.mkdir(exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*config.RECORDING_FOURCC)
            filename = recordings_dir / f"{config.RECORDING_FILENAME_PREFIX}_{int(time.time())}.mp4"
            video_writer = cv2.VideoWriter(str(filename), fourcc, config.RECORDING_FPS, frame_dims)
            is_recording = True
            print(f"Recording started: {filename}")
        else:
            is_recording = False
            if video_writer:
                video_writer.release()
            video_writer = None
            print("Recording stopped.")
    elif key == ord('s'): # 's' for screenshot
        processor.trigger_face_screenshot()
    elif key == ord('f'): # 'f' to toggle fullscreen
        is_fullscreen = not is_fullscreen
        new_state = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, new_state)
        print(f"Fullscreen mode: {'ON' if is_fullscreen else 'OFF'}")

    return should_quit, is_recording, video_writer, is_fullscreen

def _cleanup(cam, video_writer):
    """Releases all resources."""
    if video_writer:
        video_writer.release()
    cam.stop()
    cv2.destroyAllWindows()

def start_adversarial_demo(index=0):
    """Starts the fullscreen adversarial demo."""
    cam, processor, frame_dims, video_writer = _setup_application(config.CAMERA_INDEX)
    if not cam:
        return

    is_recording = False
    is_fullscreen = config.UI_FULLSCREEN

    try:
        while True:
            status, frame = cam.read()
            if not status or frame is None:
                continue
            
            frame = cv2.flip(frame, 1)
            display_frame, ui_data = processor.process_frame(frame)

            _draw_ui(display_frame, processor, ui_data, is_recording, frame_dims)
            
            if is_recording and video_writer is not None:
                video_writer.write(display_frame)

            cv2.imshow(config.WINDOW_NAME, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            should_quit, is_recording, video_writer, is_fullscreen = _handle_key_press(
                key, processor, is_recording, video_writer, frame_dims, is_fullscreen
            )
            if should_quit:
                break
    finally:
        _cleanup(cam, video_writer)

if __name__ == "__main__":
    start_adversarial_demo(index=0)