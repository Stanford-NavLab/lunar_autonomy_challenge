import numpy as np
from PIL import Image
from lang_sam import LangSAM
import torch


class Segmentation:
    def __init__(self):
        self.model = LangSAM()

    def segment_rocks(self, image: Image):
        """
        image : RGB PIL Image
        """
        text_prompt = "rock."
        results = self.model.predict([image], [text_prompt])

        full_mask = np.zeros_like(image, dtype=np.uint8).copy()
        for mask in results[0]["masks"]:
            full_mask[mask.astype(bool)] = 255

        return results[0], full_mask
