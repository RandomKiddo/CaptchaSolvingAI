# This project is licensed by the GNU GPLv3 License
# Copyright © 2023 RandomKiddo, Nishanth-Kunchala, danield33

import pandas as pd
import os
from PIL import Image


def rename_level_captchas() -> None:
    """
    Rename captchas with obscure filenames to their result
    :return: None
    """

    for _ in range(3, 6):
        csv = pd.read_csv('FILEPATH'.format(_))
        for fn in os.listdir('FILEPATH'.format(_)):
            if 'file' not in str(fn):
                continue
            os.rename('FILEPATH'.format(_, fn),
                      'FILEPATH'.format(_, f"{int(csv[csv['file_name'] == fn]['label'])}.png"))


def upsize_small_images() -> None:
    """
    Upsize all images to 256x256
    :return: None
    """

    subdir = list()
    for _, subdir, _ in os.walk('FILEPATH'):
        subdir = list(subdir)
        break
    subdir.remove('SOME_DATASET')
    for _ in subdir:
        for fn in os.listdir('FILEPATH'.format(_)):
            if fn.startswith('.'):
                continue
            img = Image.open('FILEPATH'.format(_, fn))
            img = img.resize((256, 256))
            img.save('FILEPATH'.format(_, fn))


if __name__ == '__main__':
    pass
