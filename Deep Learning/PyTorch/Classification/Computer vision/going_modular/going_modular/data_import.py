import os
import requests
import zipfile
import tarfile
from pathlib import Path

def get_data(url, file):
    '''
    Download and extract data

    Args:
        url: url to download data from
        file: name of zip file

    Returns:
        image_path: path to image folder
    '''
    # Setup path to data folder
    data_path = Path("data/")
    filename = url.split("/")[-1]
    dest_path = data_path / file
    image_path = dest_path / filename

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        dest_path.mkdir(parents=True, exist_ok=True)

        # Download data
        with open(image_path, "wb") as f:
            request = requests.get(url)
            print("Downloading data from", url)
            f.write(request.content)

        # Unzip data
        if zipfile.is_zipfile(image_path):
            print("Detected zip file -> extracting data...")
            with zipfile.ZipFile(image_path, "r") as zip_ref:
                zip_ref.extractall(dest_path)

        # Untar data
        elif tarfile.is_tarfile(image_path):
            print("Detected tar file -> extracting data...")
            with tarfile.open(image_path, "r:*") as tar_ref:
                tar_ref.extractall(dest_path)

        else:
            raise ValueError(f"File {image_path} is not a zip or tar file")

        # Remove zip file
        os.remove(image_path)

        return dest_path
