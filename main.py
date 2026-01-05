import logging
import os
import argparse
from snapmemories.core import download_memories

# import the main code

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename='', filemode='a')
logger = logging.getLogger(__file__)

def main():

    """
    Just parse the arguments and send them on to the main function
    """

    parser = argparse.ArgumentParser(
        description='Download memories from a snapchat data dump and apply time and GPS metadata and reconcile overlays')
    
    # argument 1 (required) input file type
    parser.add_argument("--input_type",
                        choice=['json', 'html'],
                        help='Specify the input file type, json or html. Defaults to html.',
                        default='html')
    
    # argument 2 (required) input file path
    parser.add_argument("-i", "input_path",
                        type=str,
                        help="Specify the input file path.")
    
    # argument 3 (required) output directory
    parser.add_argument("-o", "--output_dir",
                        type=str,
                        help="Path to where the files will be downloaded.")
    
    # argument 4 (optional) pick up
    parser.add_argument("-p", "--pickup",
                        action='store_true',
                        help="Do you want to pick up where a previous run left off")
    
    # argument 5 (optional) pick up file
    parser.add_argument("-f", "--pickup_file",
                        type=str,
                        help="Where should we pick up from?")
    

    args = parser.parse_args()

    # gotta check the input file actually exists
    if not os.path.exists(args.input_path):
        parser.error("Couldn't find the source file")
        return
    
    # check the output directory exists by making it
    os.makedirs(args.output_dir, exist_ok=True)

    # the pick up is on check that a pickup file has been specified or even exists
    if args.pickup:
        if not args.pickup_file:
            parser.error("--pickup requires a --pickup_file to be specified")
            return
        if not os.path.exists(args.pickup_file):
            parser.error("No pickup file found")
            return
        
    logger.info("STARTING")
    logger.info(f"Using a {args.input_path} file")
    logger.info(f"Saving to {args.output_dir}")
    if args.pickup:
        logger.info(f"Using {args.pickup_file} as a pickup file")

    download_memories(args.input_type, args.input_path, args.output_dir, args.pickup, args.pickup_file)

if __name__ == "__main__":
    main()