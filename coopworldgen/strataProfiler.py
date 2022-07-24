from dataclasses import dataclass
from coopworldgen.enums import *
import numpy as np

STRATA_BIOME_MAP = {
    StrataType.OCEANIC: [BiomeType.OCEAN, BiomeType.BEACH],
    StrataType.FLATLANDS: [BiomeType.FOREST, BiomeType.JUNGLE, BiomeType.DESERT, BiomeType.LAKE, BiomeType.SAVANNAH, BiomeType.SNOW],
    StrataType.HILLS: [BiomeType.FOREST, BiomeType.JUNGLE, BiomeType.DESERT, BiomeType.SNOW],
    StrataType.MOUNTAINOUS: [BiomeType.TUNDRA, BiomeType.ROCKY, BiomeType.SNOW]
}

@dataclass(frozen=True)
class StrataBiomeProfiler:


    def biome_profile(self, ) -> np.ndarray:
