import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

class VideoDataset:
    def __init__(self, video_path, max_frames=75, resize=(140, 46)):
        self.video_path = video_path
        self.max_frames = max_frames
        self.resize = resize
        self.mouth_indices = list(range(61, 88))  # MediaPipe mouth landmarks

    def _extract_mouth(self, image, landmarks):
        h, w, _ = image.shape
        points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in self.mouth_indices]
        
        x_min = min(p[0] for p in points)
        x_max = max(p[0] for p in points)
        y_min = min(p[1] for p in points)
        y_max = max(p[1] for p in points)

        pad = 10  # Optional padding
        x_min = max(0, x_min - pad)
        x_max = min(w, x_max + pad)
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad)

        mouth_roi = image[y_min:y_max, x_min:x_max]
        return mouth_roi

    def _load_video(self):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
        cap = cv2.VideoCapture(self.video_path)
        frames = []

        while len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                mouth = self._extract_mouth(frame, landmarks)
                if mouth.size == 0:
                    continue
                gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, self.resize)
                frames.append(resized)

        cap.release()
        face_mesh.close()

        while len(frames) < self.max_frames:
            frames.append(np.zeros(self.resize, dtype=np.uint8))

        video = np.stack(frames, axis=0)  # (T, H, W)
        video = video[..., np.newaxis].astype(np.float32) / 255.0  # (T, H, W, 1)
        return video

    def load(self, batch_size=2):
        video = self._load_video()
        dataset = tf.data.Dataset.from_tensor_slices([video])
        dataset = dataset.batch(batch_size)
        return dataset
