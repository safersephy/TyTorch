import tarfile
import zipfile
from pathlib import Path
from typing import Optional
from typing import Iterator, List, Optional, Tuple
import requests  # type: ignore
import torch
from loguru import logger
from tqdm import tqdm
import numpy as np
from PIL import Image

def check_create_folder(folder: Path):
    if not folder.exists():
        folder.mkdir(parents=True)


def get_file(
    data_dir: Path,
    filename: Path,
    url: str,
    unzip: bool = True,
    overwrite: bool = False,
    headers: Optional[dict] = None,
) -> Path:
    """Download a file from url to location data_dir / filename

    Args:
        data_dir (Path): dir to store file
        filename (Path): filename
        url (str): url to obtain filename
        unzip (bool, optional): If the file needs unzipping
        overwrite (bool, optional): overwrite file, if it already exists.

    Returns:
        Path: _description_
    """

    path = data_dir / filename
    if path.exists() and not overwrite:
        logger.info(f"File {path} already exists, skip download")
        return path
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}")
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 2**10
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, colour="#1e4706"
    )
    logger.info(f"Downloading {path}")
    with open(path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if unzip:
        extract(path)
    return path


def extract(path: Path) -> None:
    if path.suffix in [".zip"]:
        logger.info(f"Unzipping {path}")
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(path.parent)

    if path.suffix in [".tgz", ".tar.gz", ".gz"]:
        logger.info(f"Unzipping {path}")
        with tarfile.open(path, "r:gz") as tar:
            tar.extractall(path=path.parent)

def load_image(path: Path, image_size: Tuple[int, int]) -> np.ndarray:
    # load file
    img_ = Image.open(path).resize(image_size, Image.LANCZOS)
    return np.asarray(img_)


def split_train_val_direct(data, val_fraction):
    """
    Splits the training data and labels directly without creating an intermediate dataset.

    Args:
        data (dict): The original data loaded from the .pt file.
        val_fraction (float): Fraction of the data to use for validation.

    Returns:
        new_data (dict): New dataset containing training, validation, and test data and labels.
    """
    # Load the train data and labels
    train_data = data["traindata"]
    train_labels = data["trainlabels"]

    # Shuffle the indices
    num_samples = len(train_data)
    indices = torch.randperm(num_samples)

    # Determine split sizes
    val_size = int(num_samples * val_fraction)
    train_size = num_samples - val_size

    # Split the indices into training and validation sets
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create the train and validation splits using the indices
    train_data_split = train_data[train_indices]
    train_labels_split = train_labels[train_indices]
    val_data_split = train_data[val_indices]
    val_labels_split = train_labels[val_indices]

    # Create the new data dictionary with added validation data
    new_data = {
        "traindata": train_data_split,
        "trainlabels": train_labels_split,
        "validdata": val_data_split,
        "validlabels": val_labels_split,
        "testdata": data["testdata"],
        "testlabels": data["testlabels"],
    }

    return new_data

def walk_dir(path: Path) -> Iterator:
    """loops recursively through a folder

    Args:
        path (Path): folder to loop trough. If a directory
            is encountered, loop through that recursively.

    Yields:
        Generator: all paths in a folder and subdirs.
    """

    for p in Path(path).iterdir():
        if p.is_dir():
            yield from walk_dir(p)
            continue
        # resolve works like .absolute(), but it removes the "../.." parts
        # of the location, so it is cleaner
        yield p.resolve()

def iter_valid_paths(
    path: Path, formats: List[str]
) -> Tuple[Iterator, List[str]]:
    """
    Gets all paths in folders and subfolders
    strips the classnames assuming that the subfolders are the classnames
    Keeps only paths with the right suffix


    Args:
        path (Path): image folder
        formats (List[str]): suffices to keep, eg [".jpg", ".png"]

    Returns:
        Tuple[Iterator, List[str]]: _description_
    """
    # gets all files in folder and subfolders
    walk = walk_dir(path)
    # retrieves foldernames as classnames
    class_names = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
    # keeps only specified formats
    formats_ = [f for f in formats]
    paths = (path for path in walk if path.suffix in formats_)
    return paths, class_names