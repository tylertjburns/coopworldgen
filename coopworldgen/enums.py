from cooptools.coopEnum import CoopEnum, auto
from cooptools.colors import Color

class BiomeType(CoopEnum):
    OCEAN = Color.DARK_BLUE
    LAKE = Color.BLUE
    BEACH = Color.SANDY_BROWN
    FOREST = Color.FOREST_GREEN
    JUNGLE = Color.DARK_GREEN
    SAVANNAH = Color.TAN
    PLAINS = Color.WHEAT
    ROCKY = Color.DARK_GREY
    TUNDRA = Color.DARK_KHAKI
    DESERT = Color.BLACK
    SNOW = Color.SNOW

class StrataType(CoopEnum):
    OCEANIC = Color.DARK_BLUE
    FLATLANDS = Color.GREEN
    HILLS = Color.BROWN
    MOUNTAINOUS = Color.DARK_GREY
