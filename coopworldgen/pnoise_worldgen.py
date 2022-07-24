from typing import Tuple, Dict
import numpy as np
import coopworldgen.pnoise_utils as pn_util
import random as rnd
from coopworldgen.enums import *

def height_map(shape: Tuple[int, int],
               wavelength: Tuple[float, float],
               octaves: int,
               persistence: float,
               lacunarity: int,
               normalization: Tuple[float, float] = None,
               pnoise_offset: Tuple[float, float, float] = None,
               seed: int = None
               ) -> np.ndarray:
        if seed: rnd.seed(seed)

        if pnoise_offset is None:
            pnoise_offset = (rnd.random(), rnd.random(), rnd.random())

        return pn_util.pnoise_field(
            dims=shape,
            wavelength=wavelength,
            offset=pnoise_offset,
            octaves=octaves,
            persistence=persistence,
            lacunarity=lacunarity,
            normalization=normalization
        )


def strata_value_calculator(val) -> StrataType:
    if val < 0.25:
        return StrataType.OCEANIC
    elif val < 0.5:
        return StrataType.FLATLANDS
    elif val < 0.75:
        return StrataType.HILLS
    else:
        return StrataType.MOUNTAINOUS

def strata_map(height_arr: np.ndarray) -> np.ndarray:
    norm_01 = pn_util.normalize_between(height_arr, 0,1)

    strata_map = np.ndarray(shape=norm_01.shape, dtype=StrataType)

    for idx, val in np.ndenumerate(norm_01):
        strata_map[idx] = strata_value_calculator(val)

    return strata_map

def biome_value_calculator(height, noise_val, strata):

    if strata == StrataType.OCEANIC:
        if height < 0.2:
            return BiomeType.OCEAN
        elif height < 0.25:
            return BiomeType.BEACH
    elif strata == StrataType.FLATLANDS:
        if noise_val < 0.2:
            return BiomeType.LAKE
        elif (noise_val < 0.3):
            return BiomeType.FOREST
        elif (noise_val < 0.5):
            return BiomeType.JUNGLE
        elif (noise_val < 0.7):
            return BiomeType.SAVANNAH
        elif noise_val < 0.9:
            return BiomeType.PLAINS
        else:
            return BiomeType.DESERT
    elif strata == StrataType.HILLS:
        if noise_val < 0.3:
            return BiomeType.ROCKY
        elif noise_val < 0.5:
            return BiomeType.FOREST
        elif (noise_val < 0.8):
            return BiomeType.JUNGLE
        elif (noise_val < 0.9):
            return BiomeType.DESERT
        else:
            return BiomeType.SNOW
    elif strata == StrataType.MOUNTAINOUS:
        if (height < 0.9):
             return BiomeType.TUNDRA
        else:
            return BiomeType.SNOW

    raise NotImplementedError(f"Unhandled strata: {strata}, height: {height}, noise: {noise_val}")

def strata_noise_map(shape: Tuple[int, int], strata_biome_wavelength_map: Dict[StrataType, float]) -> Dict[StrataType, np.ndarray]:
    strata_noise = {}
    for strata in StrataType:
        strata_noise[strata] = pn_util.pnoise_field(
            dims=shape,
            wavelength=(strata_biome_wavelength_map[strata], strata_biome_wavelength_map[strata]),
            offset=(rnd.random(), rnd.random(), rnd.random()),
            octaves=1,
            persistence=0.5,
            lacunarity=2,
            normalization=(0, 1)
        )

    return strata_noise

def biome_map(height_arr: np.ndarray,
              strata_map: np.ndarray,
              strata_noise_map: Dict[StrataType, np.ndarray]) -> np.ndarray:
    norm_01_height = pn_util.normalize_between(height_arr, 0,1)

    biome_map = np.ndarray(shape=norm_01_height.shape, dtype=BiomeType)

    for idx, val in np.ndenumerate(norm_01_height):
        strata = strata_map[idx]
        noise_map = strata_noise_map[strata]
        biome_map[idx] = biome_value_calculator(height_arr[idx], noise_map[idx], strata)

    return biome_map

def threshold_eval(arr, thresh):
    return np.where(arr > thresh, np.ones(shape=arr.shape), np.zeros(shape=arr.shape))

def color_array_as_rgb(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = np.zeros(shape=arr.shape)
    g = np.zeros(shape=arr.shape)
    b = np.zeros(shape=arr.shape)

    for idx, val in np.ndenumerate(arr):
        r[idx] = val.value.value[0]
        g[idx] = val.value.value[1]
        b[idx] = val.value.value[2]

    return r, g, b

def render_color_array(name, color_array):
    r, g, b = color_array_as_rgb(color_array)
    bgr = np.dstack([b/255, g/255, r/255])
    # show
    cv2.imshow(name, bgr)

if __name__ == "__main__":
    import cv2
    np.set_printoptions(suppress=True)
    shape = (500, 750)
    wavelength = (100, 250)
    persistence=0.5
    lacunarity=2

    height_map = height_map(
        shape=shape,
        wavelength=wavelength,
        octaves=3,
        persistence=persistence,
        lacunarity=lacunarity,
        normalization=(0, 1)
    )

    strata_noise_map = strata_noise_map(shape=height_map.shape,
                                        strata_biome_wavelength_map={x: wavelength[0] / 2 for x in StrataType})

    strata = strata_map(height_arr=height_map)
    biome = biome_map(height_map, strata_map=strata, strata_noise_map=strata_noise_map)


    # show
    render_color_array("strata", strata)
    render_color_array("biome", biome)
    height_img = pn_util.greyscale_image_of_numpy_array(height_map)
    cv2.imshow('Height', height_img)
    cv2.waitKey(0)