import time

import numpy as np
from noise import pnoise3
from typing import Tuple
import cv2

def pnoise_field(dims: Tuple[int, int],
                 wavelength: Tuple[float, float],
                 offset: Tuple[float, float, float],
                 octaves: int,
                 persistence: float,
                 lacunarity: int,
                 normalization: Tuple[float, float] = None) -> np.ndarray:
    """
    :param dims: specify the number of rows/cols in the output array
    :param wavelength: tuple describing the distance covered on the perlin set (rows, cols)
    :param offset: number of iterations of layers of noise
    :param octaves: number of iterations of layers of noise
    :param persistence:
    :param lacunarity:
    :return: the perlin array defined in the space provided. Relies on the pnoise3 function
    """
    ret = np.zeros(dims, dtype=float)

    for ii in range(0, dims[0]):
        for jj in range(0, dims[1]):
            ret[ii, jj] = pnoise3(
                (ii) / wavelength[0] + offset[0],
                (jj) / wavelength[1] + offset[1],
                offset[2],
                octaves=octaves,
                persistence=persistence,
                lacunarity=lacunarity,
                repeatx=1024,
                repeaty=1024,
                base=0)

    if normalization:
        ret = normalize_between(ret, normalization[0], normalization[1])

    return ret



def normalize_between(arr: np.ndarray, min: float = 0, max: float = 1) -> np.ndarray:
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr)) * (max - min) + min

def greyscale_image_of_numpy_array(arr: np.ndarray):
    arr = normalize_between(arr)
    uint_img = np.array(arr * 255, dtype=np.uint8)

    # grayImage = cv2.adaptiveThreshold(uint_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
    return uint_img


def loop_create_perlin(
    dims: Tuple[int, int],
    wavelength: Tuple[float, float],
    start_offset: Tuple[float, float, float],
    per_frame_offset: Tuple[float, float, float],
    octaves: int,
    persistence: float,
    lacunarity: int,
    normalization: Tuple[float, float]=None
):
    current_offset = (0.0, 0.0, 0.0)
    name = 'image'
    while True:

        pnoise_array = pnoise_field(
            dims=dims,
            wavelength=wavelength,
            offset=tuple(map(sum, zip(current_offset, start_offset))),
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            normalization=normalization
        )
        frame = greyscale_image_of_numpy_array(pnoise_array)
        cv2.imshow(name, frame)
        current_offset = tuple(map(sum, zip(current_offset, per_frame_offset)))
        k = cv2.waitKey(1)
        if k == 27 or cv2.getWindowProperty(name, 0) < 0:  # wait for ESC key to exit and terminate progra,
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":

    dims=(500, 1000)
    wavelength=(10, 15)
    offset=(32, 64, 128)
    octaves=4
    persistence=0.5
    lacunarity=2
    normalization = [50, 100]

    loop_create_perlin(
        dims=dims,
        wavelength=wavelength,
        start_offset=offset,
        per_frame_offset=(0.1, 0.1, 0),
        octaves=octaves,
        persistence=persistence,
        lacunarity=lacunarity,
        normalization=normalization
    )