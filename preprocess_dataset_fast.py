import json
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
import argparse

from cropping import FaceCropper
from keypoint_detector import KeypointDetectorInsightface
from masking import Masking


# =====================================================
# PREPROCESSOR (GPU)
# =====================================================

class Preprocessor:
    def __init__(self):
        self.keypoint_detector = KeypointDetectorInsightface()
        self.face_cropper = FaceCropper()
        self.masking = Masking()

    def __call__(self, images, target_size):
        keypoints = self.keypoint_detector(images)
        cropped_images, cams, _ = self.face_cropper(
            images, keypoints, target_size
        )
        masks = self.masking(cropped_images)

        return dict(
            masks=masks["masks"],
            cams=cams,
            cropped_images=cropped_images,
        )


# =====================================================
# LOADER
# =====================================================

def loader_worker(file_queue, image_queue, image_folder):
    while True:
        item = file_queue.get()
        if item is None:
            image_queue.put(None)   # forward shutdown
            break

        i, file = item
        try:
            img = np.array(
                Image.open(os.path.join(image_folder, file)).convert("RGB")
            )
            image_queue.put((i, file, img))
        except Exception as e:
            print("Load failed:", file, e)


# =====================================================
# GPU WORKER
# =====================================================

def gpu_worker(image_queue, result_queue, resolution, batch_size, num_loaders):
    preprocessor = Preprocessor()
    batch = []
    finished_loaders = 0

    while True:
        item = image_queue.get()

        if item is None:
            finished_loaders += 1
            if finished_loaders == num_loaders:
                break
            continue

        batch.append(item)

        if len(batch) >= batch_size:
            process_batch(batch, preprocessor, resolution, result_queue)
            batch = []

    if batch:
        process_batch(batch, preprocessor, resolution, result_queue)

    result_queue.put(None)


def process_batch(batch, preprocessor, resolution, result_queue):
    indices, files, images = zip(*batch)

    outputs = preprocessor(list(images), resolution)

    for i in range(len(images)):
        result_queue.put(
            (
                indices[i],
                files[i],
                outputs["cropped_images"][i],
                outputs["masks"][i],
                outputs["cams"][i],
            )
        )


# =====================================================
# SAVER
# =====================================================

def saver_worker(result_queue, save_done_queue, result_folder, resolution):
    while True:
        item = result_queue.get()

        if item is None:
            save_done_queue.put(None)
            break

        i, file, cropped, mask, cam = item

        ending = file.split(".")[-1]
        new_filename = f"{i:09d}.{ending}"

        Image.fromarray(cropped).save(
            os.path.join(result_folder, str(resolution), new_filename)
        )

        Image.fromarray(mask).save(
            os.path.join(result_folder, "masks", new_filename)
        )

        save_done_queue.put((file, new_filename, cam))


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset preprocessing pipeline")
    parser.add_argument("--image_folder", type=str, default="test_dataset", help="Folder containing input images")
    parser.add_argument( "--result_folder", type=str, default="test_dataset_preprocessed", help="Output folder for processed dataset")
    parser.add_argument("--resolution",type=int,default=1024,help="Output face crop resolution")
    parser.add_argument("--batch_size",type=int,default=16,help="GPU batch size")
    args = parser.parse_args()

    image_folder = args.image_folder
    result_folder = args.result_folder
    resolution = args.resolution
    batch_size = args.batch_size
    os.makedirs(os.path.join(result_folder, str(resolution)), exist_ok=True)
    os.makedirs(os.path.join(result_folder, "masks"), exist_ok=True)

    files = sorted(os.listdir(image_folder))

    file_queue = Queue(256)
    image_queue = Queue(64)
    result_queue = Queue(64)
    save_done_queue = Queue(256)

    # loaders
    num_loaders = max(1, cpu_count() // 2)

    loaders = [
        Process(target=loader_worker,
                args=(file_queue, image_queue, image_folder))
        for _ in range(num_loaders)
    ]

    for p in loaders:
        p.start()

    gpu = Process(
        target=gpu_worker,
        args=(image_queue, result_queue,
              resolution, batch_size, num_loaders),
    )
    gpu.start()

    saver = Process(
        target=saver_worker,
        args=(result_queue, save_done_queue,
              result_folder, resolution),
    )
    saver.start()

    # feed files
    for item in enumerate(files):
        file_queue.put(item)

    for _ in loaders:
        file_queue.put(None)

    # metadata collection
    camera_dict = {"labels": []}
    filename_dict = {}
    filename_dict_rev = {}

    progress = tqdm(total=len(files))

    while True:
        item = save_done_queue.get()

        if item is None:
            break

        file, new_filename, cam = item

        filename_dict[file] = new_filename
        filename_dict_rev[new_filename] = file
        camera_dict["labels"].append(
            [new_filename, list(cam)]
        )

        progress.update(1)

    progress.close()

    # clean shutdown
    for p in loaders:
        p.join()

    gpu.join()
    saver.join()

    # save metadata
    with open(os.path.join(result_folder, "dataset.json"), "w") as f:
        json.dump(camera_dict, f)

    with open(os.path.join(result_folder, "filenames.json"), "w") as f:
        json.dump(filename_dict, f)

    with open(os.path.join(result_folder, "filenames_rev.json"), "w") as f:
        json.dump(filename_dict_rev, f)

    print("✅ Finished successfully.")