from typing import Tuple, Dict
import re
import logging
import os
import pandas as pd
from functools import partial
import requests
from tqdm import tqdm

logger = logging.getLogger(__file__)
tqdm.pandas()

def _fetch_response(link: str, get_req:bool)->requests.Response:
    """Get the contents of the URL"""
    if get_req:
        # again, mostly cribbed from snapchat-memories-adder. In retrospect this could probably have been a fork
        head={"X-Snap-Route-Tag": "mem-dmd", "User-Agent": "Mozilla/5.0"}
        resp = requests.get(link, headers=head, stream=True)
    else:
        # TODO
        print(f"error with {link} it's a post request")
    resp.raise_for_status()
    return resp

def get_ext(resp: requests.Response)->str:
    """Figure out what kind of file it is"""
    # again, cribbed from snapchat-memories-downloader. Figured out that the filename is in the content-disposition header
    content_disp = resp.headers.get("Content-Disposition")
    extension = ""

    if content_disp:
        # Attempt to extract file name from header
        match = re.search(r'filename="?([^"]+)"?', content_disp)
        if match:
            base_fp = match.group(1)
            _, extension = os.path.splitext(base_fp)
            extension = extension.lower() if extension else ".dat"
    return extension

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


def download_dataframe_chunks(chunk:pd.DataFrame, output_dir:str, counter, error_log):

    for index, row in chunk.iterrows():
        try:
            response = _fetch_response(row["download_link"], row["is_get_request"])
            ext = get_ext(response)
            
            is_zip = (ext == ".zip")
            fp = f"{row['file_name']}{ext}"
            file_path = os.path.join(output_dir, fp)

            with open(file_path, "wb") as f:
                for stream_chunk in response.iter_content(chunk_size=8192):
                    f.write(stream_chunk)
            
            # Write results back to the chunk using the index
            chunk.at[index, 'is_zip'] = is_zip # type: ignore
            chunk.at[index, 'file_path'] = file_path # type: ignore

        except Exception as e:
            chunk.at[index, 'is_zip'] = None # type: ignore
            chunk.at[index, 'file_path'] = None # type: ignore

            # Log the specific error along with the file name/ID
            error_info = {
                "file_name": row.get("file_name", "Unknown"),
                "error": str(e),
                "link": row.get("download_link", "N/A")
            }
            error_log.append(error_info)
        
        finally:
            # Increment the shared counter by 1
            counter.value += 1
    return chunk
    
    #def download_row(row, output_dir):
    #    try:
    #        response = _fetch_response(row["download_link"], row["is_get_request"])

    #        ext = get_ext(response)

    #        is_zip = False
    #        if ext == ".zip":
    #            is_zip = True
        
    #        fp = f"{row["file_name"]}{ext}"

    #        file_path = os.path.join(output_dir, fp)

    #        with open(file_path, "wb") as f:
    #            for chunk in response.iter_content(chunk_size=8192):
    #                f.write(chunk)
            # return an updated Series
    #        return pd.Series({
    #            'is_zip': is_zip,
    #            'file_path': file_path
    #        })

    #    except:
    #        return pd.Series({
    #            'is_zip': None,
    #            'file_path': None
    #        })

    #chunk[['is_zip', 'file_path']] = chunk.progress_apply(download_row, output_dir=output_dir, axis=1, result_type="expand") #type:ignore apparently pylance freaks out because progress_apply doesn't exist statically
    #return chunk