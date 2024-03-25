from pathlib import Path
import os
from tqdm import tqdm
from PIL import Image

def is_image(file_path, verbose=False):
    """Check if the file exists and is a valid image file.

    Args:
        file_path (str): Path to the file to check.
        verbose (bool): If True, print messages about the file's validity.
    Returns:
        bool: True if the file exists and is a valid image, False otherwise.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        if verbose: print(f"File does not exist: {file_path}")
        return False

    # Attempt to open the file as an image
    try:
        with Image.open(file_path) as img:
            # If successful, print the image format (optional)
            if verbose: print(f"Valid image file with format: {img.format}")
            if img.width != 512 or img.height != 512:
                print(f"{file_path} - wrong size")
                return False
            return True
    except (IOError, SyntaxError) as e:
        if verbose: print(f"Invalid image file: {file_path}. Error: {e}")
    except Exception as e:
        if verbose: print(f"Other error when loading the image file {e}")

    return False

items = list(Path("dataset512").rglob("*.png"))

for file in tqdm(items):
    strFile = str(file)
    if not is_image(strFile):
        os.remove(strFile)