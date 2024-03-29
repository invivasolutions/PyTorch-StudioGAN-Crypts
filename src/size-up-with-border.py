from PIL import Image, ImageOps
from pathlib import Path
from tqdm import tqdm
import os

def add_border(image_path, border_size=6):
    """
    Adds a white border to the image and saves it over the original file.
    
    :param image_path: Path object to the input image.
    :param border_size: Border size to add to each side of the image. Default is 6 pixels.
    """
    # Open the input image
    img = Image.open(str(image_path))
    
        
    # Adding border
    destPath = str(image_path).replace("dataset","dataset512")
    os.makedirs(os.path.dirname(destPath), exist_ok=True)
    
    if img.width != 512:
        new_img = ImageOps.expand(img, border=border_size, fill='white')
        new_img.save(destPath)
    else:
        img.save(destPath)

def process_images(directory_path):
    """
    Process all images in the specified directory and subdirectories, adding a border to each.
    
    :param directory_path: Path to the directory containing images.
    """
    image_folder = Path(directory_path)
    image_paths = list(image_folder.rglob('*.png'))
    
    for image_path in tqdm(image_paths, desc="Processing Images"):
        add_border(image_path)
        
# Example usage
# Replace 'your_folder_path_here' with the path to your images folder.
process_images('small_dataset/')
