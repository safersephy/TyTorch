import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import requests  # type: ignore
import tqdm
from loguru import logger


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
