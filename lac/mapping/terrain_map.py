"""Terrain map class"""

import numpy as np


class TerrainMap:
    """Terrain map class"""

    def __init__(self, resolution=0.1):
        """
        resolution : float - Map resolution in meters per pixel
        """
        self.resolution = resolution
        self.map = None
        self.map_size = None
        self.origin = None
