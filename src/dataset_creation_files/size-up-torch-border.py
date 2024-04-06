import torch
from torchvision.io import read_image, write_jpeg
from torchvision.transforms.functional import pad
from pathlib import Path
from tqdm import tqdm
import os

def add_border_to_batch(images, border_size=6):
    """
    Adds a white border to a batch of images.

    :param images: Tensor representing a batch of images.
    :param border_size: Border size to add to each side of the images. Default is 6 pixels.
    """
    # Assuming images is a tensor of shape (B, C, H, W)
    B, C, H, W = images.shape
    
    # Create a new tensor for images with borders
    new_H, new_W = H + border_size * 2, W + border_size * 2
    bordered_images = torch.full((B, C, new_H, new_W), fill_value=255)  # Fill with white

    # Place original images in the center
    bordered_images[:, :, border_size:H+border_size, border_size:W+border_size] = images
    
    return bordered_images

def process_images_in_batches(directory_path, batch_size=32):
    """
    Process all images in the specified directory in batches, adding borders to each.

    :param directory_path: Path to the directory containing images.
    :param batch_size: Number of images to process in each batch.
    """
    image_folder = Path(directory_path)
    image_paths = list(image_folder.rglob('*.png'))
    image_paths.reverse()
    device = "cuda"
    
    batch_paths = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]
    
    for batch in tqdm(batch_paths, desc="Processing Batches"):
        # Load batch
        images = torch.stack([read_image(str(path)).float() for path in batch]).to(device)
        
        # Add border
        bordered_images = add_border_to_batch(images).byte()
        
        # Save images
        for i, path in enumerate(batch):
            write_jpeg(bordered_images[i].cpu(), str(path))

# Example usage
process_images_in_batches('dataset/', batch_size=64)
