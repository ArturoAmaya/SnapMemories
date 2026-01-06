from typing import Tuple, Dict
import re
import logging
import os
import pandas as pd

logger = logging.getLogger(__file__)


def extract_coordinates(coords: str)->Tuple[float,float]:
    """
    Extract the lat and long from the combined coordinates string
    
    :param coords: Description
    :type coords: str
    :return: Description
    :rtype: Tuple[float, float]
    """
    match = re.search(r"Latitude, Longitude: (-?\d+\.?\d*), (-?\d+\.?\d*)", coords)
    if not match:
        logger.error("Bad coordinate format")
        lat, lon = 0.0, 0.0
    else:
        lat = float(match.group(1))
        long = float(match.group(2))

    return lat, long

def get_downloaded_files(output_dir:str)->Dict[str, str]:
    """
    Go to the download folder and check what's in there
    
    :param output_dir: Description
    :type output_dir: str
    :return: Description
    :rtype: Dict[str, str]
    """
    downloaded = {}
    for fp in os.listdir(output_dir):
        # check if its not a system file
        if not (fp.startswith(".") or fp.startswith("__MACOSX")):
            # do stuff
            file, _ = os.path.split(fp)
            downloaded[file] = os.path.join(output_dir, fp)
    return downloaded

def download_dataframe_chunks(chunk:pd.DataFrame):
    chunk.apply()