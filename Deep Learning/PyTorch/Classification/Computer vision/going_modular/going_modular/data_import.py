import os
import requests
import zipfile
from pathlib import Path

def get_data(url, file):
    '''
    Download and extract data

    Args:
        url: url to download data from
        file: name of zip file
    '''
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / file

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(image_path):
        # Download data
        with open(f"{image_path}.zip", "wb") as f:
            request = requests.get(url)
            print("Downloading data from", url)
            f.write(request.content)

        # Unzip data
        with zipfile.ZipFile(f"{image_path}.zip", "r") as zip_ref:
            print("Unzipping data...") 
            zip_ref.extractall(image_path)

        # Remove zip file
        os.remove(f"{image_path}.zip")
