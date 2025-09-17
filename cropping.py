import os
import cv2
import numpy as np
import urllib.request

from PIL import Image

from ddfa_v2.FaceBoxes import FaceBoxes
from ddfa_v2.TDDFA import TDDFA
from ddfa_v2.utils_3ddfa.pose import P2sRt
from crop_utils import (
    get_crop_bound,
    crop_image,
    crop_final,
    find_center_bbox,
    eg3dcamparams,
)
from keypoint_detector import KeypointDetectorInsightface


class FaceCropper:
    def __init__(self):
        base_url = "https://huggingface.co/Fubei/splatviz_inversion_checkpoints/resolve/main/"
        os.makedirs("./models", exist_ok=True)
        required_models = dict(
            checkpoint_fp="./models/mb1_120x120.pth",
            bfm_fp="./models/bfm_noneck_v3_slim.pkl",
            param_mean_std_fp="./models/param_mean_std_62d_120x120.pkl"
        )
        for key, model in required_models.items():
            if not os.path.exists(model):
                urllib.request.urlretrieve(base_url + model.split("/")[-1], model)

        self.tddfa = TDDFA(gpu_mode=True, arch="mobilenet", **required_models)
        self.face_boxes = FaceBoxes()

    def __call__(self, images, lmx, size=512):
        results_meta = []
        cropped_images = []
        bounding_boxes = []
        for i, item in enumerate(zip(images, lmx)):
            img_orig, landmarks = item
            quad, quad_c, quad_x, quad_y = get_crop_bound(landmarks)

            bound = np.array([[0, 0], [0, size - 1], [size - 1, size - 1], [size - 1, 0]], dtype=np.float32)

            mat = cv2.getAffineTransform(quad[:3], bound[:3])
            img = crop_image(img_orig, mat, size, size)
            h, w = img.shape[:2]

            # Detect faces, get 3DMM params and roi boxes
            boxes = self.face_boxes(img)
            if len(boxes) == 0:
                print(f"No face detected")
                continue

            param_lst, roi_box_lst = self.tddfa(img, boxes)
            box_idx = find_center_bbox(roi_box_lst, w, h)

            bbox_image = img.copy()
            # Draw bounding boxes on the image
            for box in roi_box_lst:
                x1, y1, x2, y2 = [int(b) for b in box[:4]]
                cv2.rectangle(bbox_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            bounding_boxes.append(bbox_image)

            param = param_lst[box_idx]
            P = param[:12].reshape(3, -1)  # camera matrix
            s_relative, R, t3d = P2sRt(P)


            # Adjust z-translation in object space
            shift_z = False
            if shift_z:
                R_ = param[:12].reshape(3, -1)[:, :3]
                u = self.tddfa.bfm.u.reshape(3, -1, order="F")
                trans_z = np.array([0, 0, 0.5 * u[2].mean()])  # Adjust the object center
                trans = np.matmul(R_, trans_z.reshape(3, 1))
                t3d += trans.reshape(3)

            """ Camera extrinsic estimation for GAN training """
            # Normalize P to fit in the original image (before 3DDFA cropping)
            x1, y1, x2, y2 = roi_box_lst[0]
            scale_x = (x2 - x1) / self.tddfa.size
            scale_y = (y2 - y1) / self.tddfa.size
            t3d[0] = (t3d[0] - 1) * scale_x + x1
            t3d[1] = (self.tddfa.size - t3d[1]) * scale_y + y1
            t3d[0] = (t3d[0] - 0.5 * (w - 1)) / (0.5 * (w - 1))  # Normalize to [-1,1]
            t3d[1] = (t3d[1] - 0.5 * (h - 1)) / (0.5 * (h - 1))  # Normalize to [-1,1], y is flipped for image space
            t3d[1] *= -1
            t3d[2] = 0
            # orthogonal camera is agnostic to Z (the model always outputs 66.67)

            s_relative = s_relative * 2000
            scale_x = (x2 - x1) / (w - 1)
            scale_y = (y2 - y1) / (h - 1)
            s = (scale_x + scale_y) / 2 * s_relative

            quad_c = quad_c + quad_x * t3d[0]
            quad_c = quad_c - quad_y * t3d[1]
            quad_x = quad_x * s
            quad_y = quad_y * s
            c, x, y = quad_c, quad_x, quad_y
            quad = np.stack([
                c - x - y,
                c - x + y,
                c + x + y,
                c + x - y
            ]).astype(np.float32)

            # final projection matrix
            s = 1
            t3d = 0 * t3d
            R[:, :3] = R[:, :3] * s
            P = np.concatenate([R, t3d[:, None]], 1)
            P = np.concatenate([P, np.array([[0, 0, 0, 1.0]])], 0)
            results_meta.append(eg3dcamparams(P.flatten()))

            # Save cropped images
            cropped_img = crop_final(img_orig, size=size, quad=quad)
            cropped_images.append(cv2.resize(cropped_img, (size, size)))

        return cropped_images, results_meta, bounding_boxes


if __name__ == "__main__":
    detector = KeypointDetectorInsightface()
    image_path = "test_files/test_face.jpg"
    image = Image.open(image_path)
    image_np = np.array(image)
    landmarks = detector([image_np])

    cropper = FaceCropper()
    cropped_images, results_meta = cropper([image_np], landmarks)
    output_path = "test_files/test_face_cropped.png"
    Image.fromarray(cropped_images[0]).save(output_path)
    print(results_meta[0])
