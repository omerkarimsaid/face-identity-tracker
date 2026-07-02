import cv2
import numpy as np
import face_recognition
from deepface import DeepFace
import time
from collections import deque, Counter
import os
import sys

# --- Config ---
REF_IMAGE = "assets/reference_face.jpeg"
MY_NAME = "john doe"
TOLERANCE = 0.50
DOWNSCALE = 0.5
EMOTION_EVERY = 2

EMOTION_MAP = {
    "happy": "happy", "sad": "sad", "angry": "sad",
    "fear": "sad", "disgust": "sad",
    "surprise": "normal", "neutral": "normal"
}

def align_face(img, box):
    top, right, bottom, left = box
    h, w = img.shape[:2]

    pad_y, pad_x = int((bottom - top) * 0.4), int((right - left) * 0.4)
    p_top, p_bottom = max(0, top - pad_y), min(h, bottom + pad_y)
    p_left, p_right = max(0, left - pad_x), min(w, right + pad_x)

    region = img[p_top:p_bottom, p_left:p_right]
    if region.size == 0: return None

    rel_loc = (top - p_top, right - p_left, bottom - p_top, left - p_left)
    landmarks = face_recognition.face_landmarks(
        cv2.cvtColor(region, cv2.COLOR_BGR2RGB), 
        face_locations=[rel_loc]
    )
    
    if not landmarks or "left_eye" not in landmarks[0] or "right_eye" not in landmarks[0]:
        return None

    lm = landmarks[0]
    left_eye, right_eye = np.mean(lm["left_eye"], axis=0), np.mean(lm["right_eye"], axis=0)

    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(region, rot_mat, (region.shape[1], region.shape[0]), flags=cv2.INTER_LINEAR)

    corners = np.array([
        [left - p_left, top - p_top], [right - p_left, top - p_top],
        [right - p_left, bottom - p_top], [left - p_left, bottom - p_top],
    ], dtype=np.float32)
    
    rotated_corners = (rot_mat @ np.hstack([corners, np.ones((4, 1))]).T).T
    xs, ys = rotated_corners[:, 0], rotated_corners[:, 1]
    
    margin = 0.15
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    fx1, fx2 = int(max(0, xs.min() - bw * margin)), int(min(region.shape[1], xs.max() + bw * margin))
    fy1, fy2 = int(max(0, ys.min() - bh * margin)), int(min(region.shape[0], ys.max() + bh * margin))

    if fx2 <= fx1 or fy2 <= fy1: return None
    return rotated[fy1:fy2, fx1:fx2]


def main():
    if not os.path.exists(REF_IMAGE):
        sys.exit(f"Error: Missing reference image: {REF_IMAGE}")

    print("Encoding reference face & warming up model...")
    ref_img_loaded = face_recognition.load_image_file(REF_IMAGE)
    ref_encodings = face_recognition.face_encodings(ref_img_loaded)
    
    if not ref_encodings:
        sys.exit(f"Error: No face detected in {REF_IMAGE}. The model may reject non-human or highly stylized images.")
    
    ref_encoding = ref_encodings[0]
    DeepFace.analyze(np.zeros((48, 48, 3), dtype=np.uint8), actions=["emotion"], detector_backend="skip", enforce_detection=False, silent=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("Error: Could not access the webcam. Check your hardware permissions.")
    
    frames = 0
    history = deque(maxlen=5)
    last_emotion = "normal"
    fps_times = deque(maxlen=30)

    print("Stream active. Press 'q' to exit.")

    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        small_rgb = cv2.resize(rgb, (0, 0), fx=DOWNSCALE, fy=DOWNSCALE)
        locs = face_recognition.face_locations(small_rgb, model="hog")
        
        inv = 1.0 / DOWNSCALE
        scaled_locs = [(int(t*inv), int(r*inv), int(b*inv), int(l*inv)) for (t, r, b, l) in locs]
        encs = face_recognition.face_encodings(rgb, scaled_locs)

        frames += 1

        for (top, right, bottom, left), enc in zip(scaled_locs, encs):
            if face_recognition.compare_faces([ref_encoding], enc, tolerance=TOLERANCE)[0]:
                
                if frames % EMOTION_EVERY == 0:
                    try:
                        aligned = align_face(frame, (top, right, bottom, left))
                        
                        crop = aligned if (aligned is not None and aligned.size > 0) else \
                               frame[max(0, top-20):bottom+20, max(0, left-20):right+20]
                        
                        raw = DeepFace.analyze(crop, actions=["emotion"], detector_backend="skip", enforce_detection=False, silent=True)
                        dom_emotion = raw[0]["dominant_emotion"] if isinstance(raw, list) else raw["dominant_emotion"]
                        
                        history.append(EMOTION_MAP.get(dom_emotion, "normal"))
                        last_emotion = Counter(history).most_common(1)[0][0]
                    except:
                        pass

                label, color = f"{MY_NAME} - {last_emotion}", (0, 200, 0)
            else:
                label, color = "Unknown", (0, 0, 200)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, max(top - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        fps_times.append(time.time() - t0)
        cv2.putText(frame, f"FPS: {1.0/(sum(fps_times)/len(fps_times)):.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()