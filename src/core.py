import logging
import pandas as pd
import logging
from io import StringIO
from src.utils import extract_coordinates, get_downloaded_files, download_dataframe_chunks
import re
from bs4 import BeautifulSoup
import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from functools import partial
from tqdm import tqdm
import multiprocessing

import warnings

# Filter out all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__file__)

def build_dataframe(input_type:str, input_path:str, output_dir:str, pickup:bool = False, pickup_file:str = "")->pd.DataFrame:
    
    logger.info("-" * 50)
    df = pd.DataFrame()
    # read from the pickup if it exists
    if pickup:
        logger.info("Picking up existing dataframe")
        df = pd.read_csv(pickup_file)
        # TODO: reconcile everything what needs to be done
    else:
        logger.info(f"Creating new dataframe using {input_path}")
        if input_type == 'html':
            logger.info("Using .html file")
            
            # open the file and read the html
            with open(input_path, "r") as f:
                html_stuff = f.read()
                df = pd.read_html(StringIO(html_stuff))[0]
        elif input_type == 'json':
            # TODO get a json file and test this
            logger.info("Using json file")
            # open the file and read the html
            with open(input_path, "r") as f:
                json_stuff = f.read()
                df = pd.read_html(StringIO(json_stuff))[0]
        else:
            logger.error(f"Something's gone wrong with the input: type {input_type} and path {input_path}")
        
        # rename the columns for ease of reading
        df = df.rename(columns={
            df.columns[0]: 'epoch_str',
            df.columns[1]: 'media_type',
            df.columns[2]: 'coords',
            df.columns[3]: 'download_link'
        }
        )

        # then do some processing - make the epoch time a number, correct the media type, add file name and file path columns, a zip flag, and extraction flag, split the coords into lat and long

        # make epoch time a number:
        df["timestamp"] = df["epoch_str"].apply(lambda time_str: int(pd.to_datetime(time_str, utc=True).timestamp()) * 1000)

        # correct the media type: make it all consistent and file-name friendly
        df["media_type"] = df["media_type"].apply(lambda media_str: media_str.replace(' ', '_').lower())

        # add file name
        df["file_name"] =  df.apply(lambda r: f"{r.name:05d}_{r['timestamp']}_{r['media_type']}", axis=1)

        # add file path
        df["file_path"] = None

        # add zip check
        df["zip"] = None

        # add extracted check
        df["extracted"] = None

        def _extractcoords(d):
            lat, long = extract_coordinates(d['coords'])
            d['lat'] = lat
            d['long'] = long
            return d
        
        df = df.apply(_extractcoords, axis=1)

        # cribbed this verbatim from snapchat-memories-downloader which is only for html need to make a json version
        if input_type == 'html':
            # Prepare for extraction of URL and boolean indicator
            soup = BeautifulSoup(html_stuff, "html.parser")

            # Find all table data rows (tr)
            rows = soup.find("table").find("tbody").find_all("tr") # type: ignore

            # The first row is the header, so we skip it (index 0)
            data_rows = rows[1:]

            # Regex pattern to capture the URL and the boolean value from the onclick attribute:
            pattern = r"downloadMemories\('(.*?)', this, (true|false)\);"

            # Lists to store the extracted data
            extracted_links, extracted_booleans = [], []

            # Iterate through rows and extract data
            for row in data_rows:
                # Find the <a> tag which contains the onclick attribute
                a_tag = row.find("a", onclick=True)
                if a_tag and "onclick" in a_tag.attrs:
                    onclick_content = a_tag["onclick"]
                    match = re.search(pattern, onclick_content) # type: ignore
                    if match:
                        extracted_links.append(match.group(1))
                        # Convert the extracted string boolean ("true" or "false") to a Python boolean
                        extracted_booleans.append(match.group(2) == "true")
                    else:
                        extracted_links.append(None)
                        extracted_booleans.append(None)
                else:
                    extracted_links.append(None)
                    extracted_booleans.append(None)

            # Add the new columns to the DataFrame
            df["download_link"] = extracted_links
            df["is_get_request"] = extracted_booleans
        elif input_type == 'json':
            # TODO
            pass
        
    # all done!
    logger.info("Done building dataframe")
    logger.info("-"*50)
    return df # type: ignore


def download_memories(input_type:str, input_path:str, output_dir:str, pickup:bool = False, pickup_file:str = ""):
    
    # make sure the output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # log the start
    logger.info("Starting")


    # build the dataframe
    df = build_dataframe(input_type, input_path, output_dir, pickup, pickup_file)

    # at this point all the entries have metadata, may/not be downloaded, may/not be metadata updated, may/not be overlayed
    # so check the table see if any have not been downloaded
    # first make a list of what's been downloaded and compare then to the table
    already_downloaded = get_downloaded_files(output_dir)
    logger.info(f"Already downloaded {len(already_downloaded)} files")

    for i, row in df.iterrows():
        file = row["file_name"]

        if file in already_downloaded:
            fp = already_downloaded[file]
            _, ext = os.path.splitext(os.path.basename(fp))

            # update
            df.loc[i, "file_path"] = fp # type: ignore
            df.loc[i, "is_zip"] = ext == ".zip" # type: ignore
            df.loc[i, "is_extracted"] = "extracted" in file # type: ignore

            logger.info(f"Skipping row {i}: File already downloaded: '{fp}'")
            continue

        # also check for missing download links
        if not row["download_link"]:
            logger.warning(f"No download link for row {i}")
        continue
    # ok cool now we know what's downloaded and whats not let's go grab everything that's not been downloaded and download it
    not_downloaded = df[(df["file_path"].isna()) & (df["download_link"].notna())]

    manager = multiprocessing.Manager()
    shared_counter = manager.Value('i', 0)
    total_rows = len(not_downloaded)
    shared_error_log = manager.list()

    pool_size = min(9, os.cpu_count() - 4) # type: ignore idk I don't want to use the whole machine
    chunks = np.array_split(not_downloaded, pool_size)
    

    download_worker_f = partial(download_dataframe_chunks, output_dir=output_dir)
    with ProcessPoolExecutor() as executor:
        #download_results = list(tqdm(executor.map(download_worker_f, chunks), total=len(chunks), desc="Downloading database chunks"))
        #df = pd.concat(download_results, ignore_index=True)
        # Use enumerate to get an index for each chunk to act as the worker_id
        # this straight up came from gemini I hope it's decent
        futures = [executor.submit(download_dataframe_chunks, c, output_dir, shared_counter, shared_error_log) for c in chunks] # type: ignore
    
        with tqdm(total=total_rows, desc="Total Downloads") as pbar:
            last_val = 0
            # While any worker is still running
            while any(f.running() for f in futures):
                current_val = shared_counter.value
                pbar.set_postfix({"failed": len(shared_error_log)})
                pbar.update(current_val - last_val)
                last_val = current_val
            
            # Final update to hit 100%
            pbar.update(total_rows - last_val)
    
    results = [f.result() for f in futures]
    final_df = pd.concat([df, pd.concat(results)])
    final_df.sort_index(axis=1, ascending=False, inplace=True)
    if shared_error_log:
        print(f"\n⚠️ {len(shared_error_log)} downloads failed.")
        errors_df = pd.DataFrame(list(shared_error_log))
        errors_df.to_csv("failed_downloads.csv", index=False)
        print("Detailed error log saved to failed_downloads.csv")
    print(final_df)
    final_df.to_csv("progress.csv", index=False)