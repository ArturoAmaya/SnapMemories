import logging
import pandas as pd
import logging
from io import StringIO

logger = logging.getLogger(__file__)

def build_dataframe(input_type:str, input_path:str, output_dir:str, pickup:bool = False, pickup_file:str = "")->pd.DataFrame:
    
    logger.info("-" * 50)
    df = None
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
            df.columns[4]: 'download_link'
        }
        )

        # then do some processing - make the epoch time a number, correct the media type, add file name and file path columns, a zip flag, and extraction flag, split the coords into lat and long


def download_memories(input_type:str, input_path:str, output_dir:str, pickup:bool = False, pickup_file:str = ""):
    df = build_dataframe(input_type, input_path, output_dir, pickup, pickup_file)