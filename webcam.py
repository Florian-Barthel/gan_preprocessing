import cv2
import numpy as np

from cropping import FaceCropper
from keypoint_detector import KeypointDetectorInsightface
from masking import Masking

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

keypoint_detector = KeypointDetectorInsightface(momentum=0.1)
face_cropper = FaceCropper()
masking = Masking()

while True:
    ret, frame = cap.read()

    image = np.array(frame)
    keypoints = keypoint_detector([image])
    if len(keypoints[0]) > 0:
        visualization_keypoints = image.copy()
        for idx, (x, y) in enumerate(keypoints[0]):
            cv2.circle(visualization_keypoints, (int(x), int(y)), 2, (0, 255, 0), -1)
            cv2.putText(visualization_keypoints, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    cropped_images, cams, boxes = face_cropper([image], keypoints, 512)
    # masks = masking(cropped_images)["applied_masks"]

    cv2.imshow('landmarks', visualization_keypoints)
    cv2.imshow('final cropped', cropped_images[0])
    cv2.imshow('crop for 3ddfa', boxes[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
