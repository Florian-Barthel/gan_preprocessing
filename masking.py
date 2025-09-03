import os

import gdown
from PIL import Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn

from MODNet.src.models.modnet import MODNet


class Masking:
    def __init__(self):
        self.im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.modnet = MODNet(backbone_pretrained=False)
        self.modnet = nn.DataParallel(self.modnet).cuda()

        if not os.path.exists("./models/modnet_photographic_portrait_matting.ckpt"):
            os.makedirs("./models", exist_ok=True)
            gdown.download(id="1mcr7ALciuAsHCpLnrtG_eop5-EYhbCmz", output="./models/modnet_photographic_portrait_matting.ckpt")
        self.modnet.load_state_dict(torch.load("./models/modnet_photographic_portrait_matting.ckpt"))
        self.modnet.eval()

    def resize_for_input(self, img: torch.Tensor, ref_size=512):
        _, _, im_h, im_w = img.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        img = F.interpolate(img, size=(im_rh, im_rw), mode="area")
        return img

    def unify_channels(self, img: np.ndarray):
        if len(img.shape) == 2:
            img = img[:, :, None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, 0:3]
        return img

    def __call__(self, images, background=(1.0, 1.0, 1.0)):
        masks = []
        applied_masks = []
        for img in images:
            img_h = img.shape[0]
            img_w = img.shape[1]

            img = self.unify_channels(img)
            img = Image.fromarray(img)
            img = self.im_transform(img)
            img = img[None, ...]

            img_resized = self.resize_for_input(img)
            _, _, pred_mask = self.modnet(img_resized.cuda(), True)
            _, _, im_h, im_w = img.shape
            mask = F.interpolate(pred_mask, size=(im_h, im_w), mode="area")
            mask = mask.detach().cpu().numpy()
            applied = img * mask + torch.tensor(background)[None, :, None, None] * (1 - mask)
            applied = ((applied.numpy() + 1) * 255 / 2).astype(np.uint8)
            applied_masks.append(applied[0].transpose(1, 2, 0))
            mask = cv2.resize((mask[0][0] * 255).astype(np.uint8), [img_w, img_h])
            masks.append(mask)
        return dict(masks=masks, applied_masks=applied_masks)


if __name__ == "__main__":
    masker = Masking()
    image_path = "test_files/test_face.jpg"
    image = Image.open(image_path)
    image_np = np.array(image)
    masks = masker([image_np])
    Image.fromarray(masks["masks"][0]).save("test_files/test_face_mask.png")
    Image.fromarray(masks["applied_masks"][0]).save("test_files/test_face_mask_applied.png")
