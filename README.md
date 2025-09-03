# GAN Preprocessing

This is the preprocessing pipeline of [CGS-GAN](https://github.com/fraunhoferhhi/cgs-gan).

## Install

1. Install the environment.
2. run `./build.sh` 

## Usage

```python
import numpy as np
from PIL import Image

from preprocess import Preprocessor


if __name__ == "__main__":
    preprocessor = Preprocessor()
    image_path = "test_files/test_face.jpg"
    image = np.array(Image.open(image_path))
    preprocessed = preprocessor([image], 512)
    output_path = "test_files/test_face_preprocessed.png"
    Image.fromarray(preprocessed["cropped_images"][0]).save(output_path)
    print(preprocessed["cams"][0])
    print(preprocessed["masks"][0])
```
