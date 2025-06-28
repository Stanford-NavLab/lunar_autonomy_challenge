"""Wrapper for finetuned Unet++ segmentation model."""

import numpy as np
import cv2
import torch
from torchvision.transforms.functional import to_tensor, resize
from PIL import Image
import segmentation_models_pytorch as smp
from pathlib import Path
from enum import Enum

from lac.params import TEAM_CODE_ROOT


class SemanticClasses(Enum):
    FIDUCIALS = 0
    ROCK = 1
    LANDER = 2
    GROUND = 3
    SKY = 4


class UnetSegmentation:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = Path(TEAM_CODE_ROOT) / "models" / "unet_v2.pth"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (
            smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights="imagenet",
                in_channels=3,
                classes=5,
            )
            .to(self.device)
            .to(memory_format=torch.channels_last)
        )
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.downscale_factor = 2

    def predict(self, image: np.ndarray):
        """
        Predict the segmentation mask for a given image.
        Args:
            image: Input image as a numpy array.
        Returns:
            Segmentation mask as a numpy array.
        """
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape
        H_ds = (H // self.downscale_factor) // 32 * 32
        W_ds = (W // self.downscale_factor) // 32 * 32

        img_resized = resize(Image.fromarray(img), (H_ds, W_ds))
        img_tensor = to_tensor(img_resized).to(self.device).unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)

        # TODO: get confidence values from output and return them

        pred = output.argmax(1).squeeze().cpu()
        pred_resized = cv2.resize(pred.numpy(), (W, H), interpolation=cv2.INTER_NEAREST)

        return pred_resized

    def segment_rocks(self, image: np.ndarray, output_pred=False):
        """
        Segment rocks in the input image using the trained Unet++ model.
        Args:
            image: Input image as a numpy array.
        Returns:
            List of masks for each detected rock
        """
        pred = self.predict(image)
        rock_mask = pred == SemanticClasses.ROCK.value

        # Identify unique rock masks
        num_labels, labels = cv2.connectedComponents(rock_mask.astype(np.uint8))

        MIN_ROCK_MASK_AREA = 100  # Minimum area to be considered a valid rock segmentation

        masks = []

        for label in range(1, num_labels):
            mask = labels == label
            if np.sum(mask) > MIN_ROCK_MASK_AREA:
                masks.append(mask)
            else:
                labels[mask] = 0  # Remove small masks

        if output_pred:
            return masks, labels, pred
        else:
            return masks, labels
