from typing import Tuple, Dict
import re
import logging
import os
import pandas as pd
from functools import partial
import requests
from tqdm import tqdm
import zipfile
import shutil
from tqdm.contrib.logging import logging_redirect_tqdm

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
            file, _ = os.path.splitext(fp)
            downloaded[file] = os.path.join(output_dir, fp)
    return downloaded

def _unzips(df:pd.DataFrame, output_dir:str):
    def is_system_file(file_name):
        return file_name.startswith('__MACOSX') or file_name.startswith('.')
    """Unzip the zip files"""
    # credit to snapchat-memories-downloader

    # only keep the zips
    zip_df = df[df["zip"] == True]

    unzipped = 0
    new_rows = []
    total_tounzip = len(zip_df)

    errors = []

    with logging_redirect_tqdm():
        with tqdm(total=total_tounzip, desc="Unzipping") as pbar:
            for index, row in zip_df.iterrows():
                zip_file_path = row["file_path"]
                zip_file_name = row["file_name"]
                # Use the file name (without .zip) as the temporary extraction folder name
                temp_extract_dir = os.path.join(output_dir, f"temp_{zip_file_name}")

                try:
                    # Extract the ZIP file
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        # Create the temporary directory inside the download folder
                        os.makedirs(temp_extract_dir, exist_ok=True)
                        zip_ref.extractall(temp_extract_dir)
                        logger.debug(f"Successfully extracted to: '{temp_extract_dir}'")

                    # Process extracted files, rename, and move
                    extracted_files_count = 0
                    for root, _, extractable_files in os.walk(temp_extract_dir):
                        for i, extractable_file_name in enumerate(extractable_files):
                            # Ignore macOS resource files or system files
                            if is_system_file(extractable_file_name):
                                continue

                            # Extract file path in temp directory
                            extractable_file_path = os.path.join(root, extractable_file_name)
                            _, extension = os.path.splitext(extractable_file_name)

                            # Final file name to use for extracted file (zip_filename_extracted_index+1)
                            final_extracted_base_file_path = f"{zip_file_name}_extracted_{i + 1}{extension}"
                            final_extracted_file_path = os.path.join(output_dir, final_extracted_base_file_path)

                            # Move the file to the parent download directory
                            shutil.move(extractable_file_path, final_extracted_file_path)

                            # Creating new row entry for dataframe
                            new_row = row.copy()
                            new_row["file_name"] = os.path.splitext(final_extracted_base_file_path)[0]  # without extension
                            new_row["file_path"] = final_extracted_file_path
                            new_row["is_zip"] = False
                            new_row["is_extracted"] = True  # Keeping track of extracted memories
                            new_rows.append(new_row)

                            extracted_files_count += 1

                    unzipped += 1
                    logger.info(
                        f"[{unzipped}/{total_tounzip}] Successfully moved {extracted_files_count} files to '{output_dir}'."
                    )

                except zipfile.BadZipFile as e:
                    logger.error(f"Error: The downloaded file '{zip_file_path}' is not a valid ZIP file.")
                    errors.append({"index": index, "error": str(e)})
                    pbar.set_postfix(failed=len(errors))

                except Exception as e:
                    logger.error(f"Error processing ZIP file '{zip_file_path}': {e}")
                    errors.append({"index": index, "error": str(e)})
                    pbar.set_postfix(failed=len(errors))
                finally:
                    # Clean up the temporary folder
                    if os.path.exists(temp_extract_dir):
                        shutil.rmtree(temp_extract_dir)
                        logger.debug(f"Deleted temporary folder: {temp_extract_dir}")
                    pbar.update(1)
    if errors:
        print(f"\nFinished with {len(errors)} failures.")
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
            chunk.at[index, 'zip'] = is_zip # type: ignore
            chunk.at[index, 'file_path'] = file_path # type: ignore

        except Exception as e:
            chunk.at[index, 'zip'] = None # type: ignore
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