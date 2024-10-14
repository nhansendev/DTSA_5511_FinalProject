import os
import numpy as np
from PIL import Image

SCRIPT_DIR = os.getcwd()


def make_numpy_file(data_dir, save_name):
    data_path = os.path.join(SCRIPT_DIR, data_dir)
    files = [os.path.join(data_path, f) for f in os.listdir(data_path)]

    images = []
    for f in files:
        images.append(np.asarray(Image.open(f)))

    images = np.array(images)
    # Data shape is: (batch, width, height, channels)

    np.save(os.path.join(SCRIPT_DIR, save_name + ".npy"), images)


if __name__ == "__main__":
    data_dir = "Kirmizi_Pistachio"

    make_numpy_file(data_dir, "kirmizi")
