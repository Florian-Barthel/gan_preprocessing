import cv2
import numpy as np

from keypoint_detector import KeypointDetectorInsightface
from preprocess import Preprocessor

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 512)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
preprocessor = Preprocessor()

while True:
    ret, frame = cap.read()

    image_np = np.array(frame)
    result = preprocessor([frame], target_size=512)["cropped_images"][0]

    """
    # Visualize landmarks
    if len(landmarks[0]) > 0:
        visualization = image_np.copy()
        for idx, (x, y) in enumerate(landmarks[0]):
            cv2.circle(visualization, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(visualization, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)
    """

    cv2.imshow('frame', result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
