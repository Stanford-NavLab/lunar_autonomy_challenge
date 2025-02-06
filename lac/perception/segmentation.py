import numpy as np
from PIL import Image
from lang_sam import LangSAM


class Segmentation:
    def __init__(self):
        self.model = LangSAM()

    def segment_rocks(self, image):
        text_prompt = "rock."
        results = self.model.predict([image], [text_prompt])

        image_out = np.zeros_like(image)
        for mask in results[0]["masks"]:
            image_out[mask.astype(bool)] = 255

        return results[0], image_out
