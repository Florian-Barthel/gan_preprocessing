import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from cropping import FaceCropper
from keypoint_detector import KeypointDetectorInsightface
from masking import Masking


class Preprocessor:
    def __init__(self):
        self.keypoint_detector = KeypointDetectorInsightface()
        self.face_cropper = FaceCropper()
        self.masking = Masking()

    def __call__(self, images, target_size, apply_mask=False):
        keypoints = self.keypoint_detector(images)
        cropped_images, cams, bbox = self.face_cropper(images, keypoints, target_size)
        masks = self.masking(cropped_images)
        if apply_mask:
            cropped_images = masks["applied_masks"]
        return dict(masks=masks["masks"], cams=cams, cropped_images=cropped_images)


if __name__ == "__main__":
    image_folder = "test_dataset"
    result_folder = "test_dataset_preprocessed"
    resolution = 1024
    camera_dict = {"labels": []}
    filename_dict = {}
    filename_dict_rev = {}


    os.makedirs(result_folder, exist_ok=True)
    os.makedirs(os.path.join(result_folder, str(resolution)), exist_ok=True)
    os.makedirs(os.path.join(result_folder, "masks"), exist_ok=True)

    preprocessor = Preprocessor()
    for i, file in tqdm(enumerate(sorted(os.listdir(image_folder)))):
        image = np.array(Image.open(os.path.join(image_folder, file)))
        preprocessed = preprocessor([image], resolution)

        ending = file.split(".")[-1]
        new_filename = f"{i:09d}.{ending}"
        filename_dict[file] = new_filename
        filename_dict_rev[new_filename] = file
        Image.fromarray(preprocessed["cropped_images"][0]).save(os.path.join(result_folder, str(resolution), new_filename))
        Image.fromarray(preprocessed["masks"][0]).save(os.path.join(result_folder, "masks", new_filename))
        camera_dict["labels"].append([new_filename, list(preprocessed["cams"][0])])

    with open(result_folder + "/dataset.json", "w") as f:
        f.write(json.dumps(camera_dict))

    with open(result_folder + "/filenames.json", "w") as f:
        f.write(json.dumps(filename_dict))

    with open(result_folder + "/filenames_rev.json", "w") as f:
        f.write(json.dumps(filename_dict_rev))