from typing import Optional, List
from PIL import Image
import os
import glob
from moviepy import VideoFileClip


def get_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    """
    Get all image files from a directory, sorted by name.

    Args:
        directory: Path to the directory containing images
        extensions: List of file extensions to include (e.g., ['.png', '.jpg'])

    Returns:
        List of image file paths, sorted
    """
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

    image_files = []
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        image_files.extend(glob.glob(pattern))
        pattern = os.path.join(directory, f"*{ext.upper()}")
        image_files.extend(glob.glob(pattern))

    # Sort files to ensure consistent ordering
    image_files.sort()
    return image_files


def resize_images(
    images: List[Image.Image], scale_factor: Optional[float] = None
) -> List[Image.Image]:
    """
    Resize all images by a scaling factor to maintain aspect ratio.

    Args:
        images: List of PIL Image objects
        scale_factor: Scaling factor (e.g., 0.5 for half size, 2.0 for double size). If None, use the size of the first image.

    Returns:
        List of resized images
    """
    if scale_factor is None:
        return images

    resized_images = []
    for img in images:
        if scale_factor != 1.0:
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_images.append(img)

    return resized_images


def create_gif(
    input_dir: str,
    output_path: str,
    duration: int = 500,
    loop: int = 0,
    scale_factor: Optional[float] = None,
    optimize: bool = True,
    quality: int = 85,
    extensions: List[str] = None,
) -> None:
    """
    Create a GIF from images in a directory.

    Args:
        input_dir: Directory containing the images
        output_path: Path for the output GIF file
        duration: Duration for each frame in milliseconds
        loop: Number of loops (0 for infinite)
        scale_factor: Scaling factor for resizing (e.g., 0.5 for half size, 2.0 for double size)
        optimize: Whether to optimize the GIF
        quality: Quality setting for optimization (1-100)
        extensions: List of file extensions to include
    """
    # Get image files
    image_files = get_image_files(input_dir, extensions)

    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")

    print(f"Found {len(image_files)} images in {input_dir}")

    # Load images
    images = []
    for file_path in image_files:
        try:
            img = Image.open(file_path)
            # Convert to RGB if necessary (GIF doesn't support RGBA)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            images.append(img)
            print(f"Loaded: {os.path.basename(file_path)} ({img.size})")
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")

    if not images:
        raise ValueError("No valid images could be loaded")

    # Resize images if requested
    if scale_factor:
        print(f"Resizing images by scale factor: {scale_factor}")
        images = resize_images(images, scale_factor)

    # Save as GIF
    print(f"Creating GIF: {output_path}")
    print(f"Duration: {duration}ms per frame")
    print(f"Loop: {'infinite' if loop == 0 else loop} times")

    # Save the first image and append the rest
    first_image = images[0]
    remaining_images = images[1:]

    first_image.save(
        output_path,
        save_all=True,
        append_images=remaining_images,
        duration=duration,
        loop=loop,
        optimize=optimize,
        quality=quality,
    )

    print(f"GIF created successfully: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")


def convert_gif_to_mp4(gif_path, mp4_path, fps=None):
    """NOTE: doesn't seem to be working"""
    clip = VideoFileClip(gif_path)
    if fps:
        clip = clip.set_fps(fps)
    clip.write_videofile(mp4_path, codec="libx264", audio=False)
