"""LangSAM wrapper"""

import numpy as np
from PIL import Image
from lang_sam import LangSAM

from lac.params import ROCK_MASK_MAX_AREA, ROCK_BRIGHTNESS_THRESHOLD


class LangSAMSegmentation:
    def __init__(self):
        self.model = LangSAM()

    def segment_rocks(self, image: Image):
        """
        image : RGB PIL Image
        """
        text_prompt = "rock."
        results = self.model.predict([image], [text_prompt])

        full_mask = np.zeros_like(image, dtype=np.uint8).copy()
        image_np = np.array(image.convert("L"))
        masks = []
        for mask in results[0]["masks"]:
            if (
                mask.sum() < ROCK_MASK_MAX_AREA
                and image_np[mask.astype(bool)].mean() > ROCK_BRIGHTNESS_THRESHOLD
            ):
                masks.append(mask)
                full_mask[mask.astype(bool)] = 255

        return masks, full_mask
