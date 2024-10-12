import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import requests  # type: ignore
import torch
from loguru import logger
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def add_batch_padding(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a batch of sequences and converts labels to a tensor.

    Args:
        batch: A list of tuples where each tuple contains a tensor (sequence) and an integer label.

    Returns:
        A tuple containing:
        - A tensor of padded sequences.
        - A tensor of labels.
    """
    X, y = zip(*batch)  # noqa: N806
    X_ = pad_sequence(X, batch_first=True)  # noqa: N806
    return X_, torch.tensor(y)  # noqa: N806


def check_create_folder(folder: Path) -> None:
    if not folder.exists():
        folder.mkdir(parents=True)


def get_file(
    data_dir: Path,
    filename: Path,
    url: str,
    unzip: bool = True,
    overwrite: bool = False,
    headers: Optional[Dict[str, str]] = None,
) -> Path:
    """Download a file from url to location data_dir / filename

    Args:
        data_dir (Path): dir to store file
        filename (Path): filename
        url (str): url to obtain filename
        unzip (bool, optional): If the file needs unzipping
        overwrite (bool, optional): overwrite file, if it already exists.

    Returns:
        Path: The path of the downloaded file
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


def download_data(
    source_url: str,
    bronze_folder: str,
    bronze_filename: str,
    unzip: bool = False,
    overwrite: bool = False,
) -> Optional[Path]:
    """
    Standalone function to handle the extraction and return the path to the downloaded file.

    Args:
        source_url (str): URL to download the data
        bronze_folder (str): Folder to save the downloaded file
        bronze_filename (str): Filename to save the file as
        unzip (bool): Whether to unzip the file after downloading
        overwrite (bool): Whether to overwrite the file if it already exists

    Returns:
        Path: Path of the downloaded file, or None if the file already exists and overwrite is False
    """
    # Convert bronze_folder and bronze_filename to Path objects
    bronze_folder_path = Path(bronze_folder)
    bronze_filename_path = Path(bronze_filename)

    # Create the folder if it doesn't exist
    if not bronze_folder_path.exists():
        bronze_folder_path.mkdir(parents=True)

    # Check if the file already exists
    if (bronze_folder_path / bronze_filename_path).exists() and not overwrite:
        logger.info(
            f"File {bronze_filename_path} already exists at {bronze_folder_path}"
        )
        return None

    # Download the file using the `get_file` utility
    bronze_path = get_file(
        bronze_folder_path,
        bronze_filename_path,
        url=source_url,
        unzip=unzip,
        overwrite=overwrite,
    )

    return bronze_path


def custom_split(
    data: Any,
    labels: Any,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[Tuple[Any, Any], Tuple[Any, Any], Tuple[Any, Any]]:
    """
    Custom dataset split function that handles pandas DataFrame, Polars DataFrame,
    NumPy array, and PyTorch tensor, splitting them into train, validation, and test sets.

    Args:
        data: The dataset (pandas DataFrame, Polars DataFrame, NumPy array, or PyTorch tensor).
        labels: Corresponding labels.
        test_size: Proportion of the data to be used as the test set.
        val_size: Proportion of the remaining data to be used as the validation set.
        random_state: Seed for reproducibility.

    Returns:
        Tuple containing train, validation, and test splits of data and labels.
    """
    np.random.seed(random_state)

    if isinstance(data, (pd.DataFrame, pl.DataFrame)):
        n_samples = data.shape[0]
    elif isinstance(data, (np.ndarray, torch.Tensor)):
        n_samples = data.shape[0]
    else:
        raise ValueError("Unsupported data format")

    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_size = int(n_samples * test_size)
    val_size = int((n_samples - test_size) * val_size)

    test_indices = indices[:test_size]
    val_indices = indices[test_size : test_size + val_size]
    train_indices = indices[test_size + val_size :]

    def split_data(data: Any, indices: np.ndarray) -> Any:
        if isinstance(data, pd.DataFrame):
            return data.iloc[indices]
        elif isinstance(data, pl.DataFrame):
            return data[indices]
        elif isinstance(data, np.ndarray):
            return data[indices, :]
        elif isinstance(data, torch.Tensor):
            return data[indices]
        else:
            raise ValueError("Unsupported data format")

    train_data = split_data(data, train_indices)
    val_data = split_data(data, val_indices)
    test_data = split_data(data, test_indices)

    train_labels = split_data(labels, train_indices)
    val_labels = split_data(labels, val_indices)
    test_labels = split_data(labels, test_indices)

    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)


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
    img_ = Image.open(path).resize(image_size, Image.LANCZOS)
    return np.asarray(img_)


def split_train_val_direct(data: Dict[str, Any], val_fraction: float) -> Dict[str, Any]:
    """
    Splits the training data and labels directly without creating an intermediate dataset.

    Args:
        data (dict): The original data loaded from the .pt file.
        val_fraction (float): Fraction of the data to use for validation.

    Returns:
        new_data (dict): New dataset containing training, validation, and test data and labels.
    """
    train_data = data["traindata"]
    train_labels = data["trainlabels"]

    num_samples = len(train_data)
    indices = torch.randperm(num_samples)

    val_size = int(num_samples * val_fraction)
    train_size = num_samples - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data_split = train_data[train_indices]
    train_labels_split = train_labels[train_indices]
    val_data_split = train_data[val_indices]
    val_labels_split = train_labels[val_indices]

    new_data = {
        "traindata": train_data_split,
        "trainlabels": train_labels_split,
        "validdata": val_data_split,
        "validlabels": val_labels_split,
        "testdata": data["testdata"],
        "testlabels": data["testlabels"],
    }

    return new_data


def walk_dir(path: Path) -> Iterator[Path]:
    """loops recursively through a folder

    Args:
        path (Path): folder to loop through. If a directory
            is encountered, loop through that recursively.

    Yields:
        Generator: all paths in a folder and subdirs.
    """

    for p in Path(path).iterdir():
        if p.is_dir():
            yield from walk_dir(p)
        else:
            yield p.resolve()


def iter_valid_paths(
    path: Path, formats: List[str]
) -> Tuple[Iterator[Path], List[str]]:
    """
    Gets all paths in folders and subfolders
    strips the classnames assuming that the subfolders are the classnames
    Keeps only paths with the right suffix


    Args:
        path (Path): image folder
        formats (List[str]): suffices to keep, eg [".jpg", ".png"]

    Returns:
        Tuple[Iterator[Path], List[str]]: An iterator of paths and a list of class names
    """
    walk = walk_dir(path)
    class_names = [subdir.name for subdir in path.iterdir() if subdir.is_dir()]
    formats_ = [f for f in formats]
    paths = (path for path in walk if path.suffix in formats_)
    return paths, class_names
