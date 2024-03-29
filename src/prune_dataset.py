# This will remove x% of images from the dataset at random. The default is to delete the images, but it will move
# them instead if you specify a "moveTarget" as an argument on the command line.
# the intention being to create a smaller practice dataset to try out commands on to see if it works.
# 
from pathlib import Path
import os, shutil
from tqdm.auto import tqdm
import argparse
import random

def main(move_target, elim_percent, dataDir):
    items = list(Path(dataDir).rglob("*.png")) + list(Path(dataDir).rglob("*.jpg")) + list(Path(dataDir).rglob("*.PNG"))+list(Path(dataDir).rglob("*.JPG"))
    items = list(set(items))
    
    eliminateInt = int(len(items) * elim_percent / 100)
    deleteList = random.sample(items, eliminateInt)
    for image in tqdm(deleteList):
        if move_target is None:
            os.remove(str(image))
        else:
            shutil.move(str(image),os.path.join(move_target,image.name))
    
if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some integers.")

    # Define the arguments
    parser.add_argument("--dataRoot", type=str, required=True, help="The Base Dataset Directory")
    parser.add_argument("--elimPercent", type=float, required=True, help="The percentage of the dataset to eliminate.")
    parser.add_argument("--moveTarget", type=str, default=None, help="[Optional] instead of deleting the images, move them here instead.")
    
    

    # Parse the arguments
    args = parser.parse_args()

    # Call main with the processed arguments
    main(args.moveTarget, args.elimPercent, args.dataRoot)
