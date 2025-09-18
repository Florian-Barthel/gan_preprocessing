import numpy as np
from PIL import Image

from cropping import FaceCropper
from keypoint_detector import KeypointDetectorInsightface
from masking import Masking


class Preprocessor:
    def __init__(self):
        self.keypoint_detector = KeypointDetectorInsightface()
        self.face_cropper = FaceCropper()
        self.masking = Masking()

    def __call__(self, images, target_size, apply_mask=True):
        keypoints = self.keypoint_detector(images)
        cropped_images, cams, bbox = self.face_cropper(images, keypoints, target_size)
        masks = self.masking(cropped_images)
        if apply_mask:
            cropped_images = masks["applied_masks"]
        return dict(masks=masks["masks"], cams=cams, cropped_images=cropped_images)


if __name__ == "__main__":
    preprocessor = Preprocessor()
    image_path = "test_files/test_face.jpg"
    image = np.array(Image.open(image_path))
    preprocessed = preprocessor([image], 512)
    output_path = "test_files/test_face_preprocessed.png"
    Image.fromarray(preprocessed["cropped_images"][0]).save(output_path)
    print(preprocessed["cams"][0])
    print(preprocessed["masks"][0])