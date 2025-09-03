from insightface import app
from PIL import Image
import numpy as np
import cv2


class KeypointDetectorInsightface:
    def __init__(self):
        self.app = app.FaceAnalysis(name="buffalo_l", verbose=False)
        self.app.prepare(ctx_id=0)

    def __call__(self, images):
        landmarks = []
        for img in images:
            detection = self.app.get(img)
            if len(detection) > 0:
                landmarks.append(detection[0].landmark_2d_106)
            else:
                landmarks.append([])
        return landmarks


if __name__ == "__main__":
    detector = KeypointDetectorInsightface()
    image_path = "test_files/test_face.jpg"
    image = Image.open(image_path)
    image_np = np.array(image)
    landmarks = detector([image_np])

    # Visualize landmarks
    if len(landmarks[0]) > 0:
        visualization = image_np.copy()
        for idx, (x, y) in enumerate(landmarks[0]):
            cv2.circle(visualization, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(visualization, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)
        cv2.imwrite("test_files/test_face_landmarks.png", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))

