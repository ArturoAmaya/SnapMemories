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
from datetime import datetime
from zoneinfo import ZoneInfo
from exiftool import ExifToolHelper
from timezonefinder import TimezoneFinder
import subprocess
import glob
from pathlib import Path


logger = logging.getLogger(__file__)
tqdm.pandas()

# Identify the "liars" (PNGs that are actually WebPs)
# We can do this by checking the first few bytes of the file
def is_actually_webp(file_path):
    try:
        with open(file_path, 'rb') as f:
            sig = f.read(12)
            # WebP files start with RIFF....WEBP
            return sig.startswith(b'RIFF') and b'WEBP' in sig
    except:
        return False

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

    # only keep the zips that haven't been extracted already
    zip_df = df[(df["zip"] == True) & (df["been_extracted"] == False)]

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
                            new_row["zip"] = False
                            new_row["been_extracted"] = False
                            new_row["is_an_extract"] = True  # Keeping track of extracted memories
                            new_row["metadata_updated"] = False
                            new_rows.append(new_row)

                            extracted_files_count += 1
                    zip_df.at[index, 'been_extracted'] = True #type:ignore make sure the original zip is logged as extracted in the progress dataframe
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
    for new_row in new_rows:
        df.loc[len(df)] = new_row

    logger.info(f"Added {len(new_rows)} new memories to dataframe!")
    return zip_df

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
            chunk.at[index, 'been_extracted'] = False #type: ignore
            chunk.at[index, 'is_an_extract'] = False #type: ignore
            chunk.at[index, "metadata_updated"] = False #type:ignore

        except Exception as e:
            chunk.at[index, 'zip'] = None # type: ignore
            chunk.at[index, 'file_path'] = None # type: ignore
            chunk.at[index, 'been_extracted'] = None #type: ignore
            chunk.at[index, 'is_an_extract'] = None #type: ignore
            chunk.at[index, 'metadata_updated'] = None #type:ignore


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

def get_local_time_from_location(latitude, longitude, utc_datetime):
    """
    Determines the local timezone from coordinates and converts a UTC datetime object 
    to that local time.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        utc_datetime (datetime): A timezone-aware datetime object in UTC.

    Returns:
        datetime: The datetime object converted to the local timezone.
    """
    # 1. Determine the timezone name (e.g., 'America/New_York') from coordinates
    tf = TimezoneFinder()
    timezone_name = tf.timezone_at(lng=longitude, lat=latitude)
    
    if timezone_name is None:
        return "Could not determine the timezone for the given coordinates."

    # 2. Get the timezone object using zoneinfo
    try:
        local_tz = ZoneInfo(timezone_name)
    except Exception as e:
        return f"Error creating ZoneInfo object: {e}"

    # 3. Convert the UTC datetime object to the local timezone
    # Ensure the input datetime is timezone-aware in UTC first
    if utc_datetime.tzinfo is None:
        # It's best practice to explicitly make a naive datetime UTC-aware first
        utc_datetime = utc_datetime.replace(tzinfo=ZoneInfo('UTC'))
        
    local_datetime = utc_datetime.astimezone(local_tz)

    return local_datetime, timezone_name

def _update_media_metadata_pyexiftool(file_path, timestamp_str, lat, lon):
    """Update the EXIF data using exiftool. Important difference relative to snapchat-memories-downloader is that this
    one does the timezone offset properly. Without that all the times are set to UTC which is not great when you put it into a library they show up out of order"""

    # structure taken from snapchat-memories-downloader with some structural edits
    if not os.path.exists(file_path):
        logger.error(f"File not found: '{file_path}'")
        return

    # Parse Timestamp
    try:
        # Convert timestamp '2025-11-13 22:15:16 UTC' to the required Exif format
        # make the datetime object not naive, then extract the correct offset using the timezones code
        dt_object = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S UTC")
        dt_object.replace(tzinfo=ZoneInfo('UTC'))
        local_time, tz_name = get_local_time_from_location(lat, lon, dt_object)

        exif_datetime_format = local_time.strftime("%Y:%m:%d %H:%M:%S%:z") # type:ignore lesson learned from be real exporter :cry need to include the : in z otherwise they all freak out
    except ValueError as e:
        logger.error(f"Error parsing data for '{file_path}': {e}")
        return

    # Format the coordinates as D/M/S (or decimal) string with N/S/E/W suffix for GPSCoordinates tag
    # ExifTool can handle the conversion, but combining them is best for QuickTime
    gps_coordinate_string = f"{abs(lat)} {'N' if lat >= 0 else 'S'} {abs(lon)} {'E' if lon >= 0 else 'W'}"

    # Prepare Metadata Tags
    # We use both Date/Time tags to ensure both photo (.jpg) and video (.mp4) files are covered.
    # ExifTool automatically converts the absolute lat/lon values into the required
    # degrees, minutes, seconds (DMS) format and sets the reference tags (N/S/E/W).
    metadata_tags = {
        # Time and Date tags
        "XMP:DateTimeOriginal": exif_datetime_format,  # Primary XMP tag for time
        "XMP:CreateDate": exif_datetime_format,  # Secondary XMP tag
        "DateTimeOriginal": exif_datetime_format,  # Standard EXIF tag for original time
        "CreateDate": exif_datetime_format,  # XMP/QuickTime tag (useful for MP4)
        "ModifyDate": exif_datetime_format,  # Update the file modification date

        # Ok so apparently QuickTime times are UTC so that means no -8, -5 +4 whatever. bake it in to the string, ie. use the original that we had
        "QuickTime:CreateDate": dt_object.strftime("%Y:%m:%d %H:%M:%S"),
        "QuickTime:ModifyDate": dt_object.strftime("%Y:%m:%d %H:%M:%S"),
        # hopefully these don't fuck up images. seems like this works with Google Photos without modifying anything else

        # GPS tags (ExifTool automatically calculates DMS from decimal degrees)
        "XMP:GPSLatitude": lat,
        "XMP:GPSLongitude": lon,
        "GPSLatitude": lat,
        "GPSLongitude": lon,

        # Optional: Set the reference direction explicitly if needed, but ExifTool can derive this
        # We need this to fix incorrect coordinate derivation by ExifTool
        "GPSLatitudeRef": 'N' if lat >= 0 else 'S',
        "GPSLongitudeRef": 'E' if lon >= 0 else 'W',

        # **CRITICAL for MP4 (QuickTime/XMP)** Mac and iPhone still don't show location of video! Need fix!
        "GPSCoordinates": gps_coordinate_string,  # Writes location in one tag for QuickTime/XMP
        "Location": gps_coordinate_string,  # Used by some readers
        "UserData:GPSCoordinates": f"{lat:+08.4f}{lon:+09.4f}/", # userdata and quicktime have been added to hopefully fix google photos but keys was the key one
        "QuickTime:GPSCoordinates": f"{lat:+08.4f}{lon:+09.4f}/",
        "Keys:GPSCoordinates": f"{lat:+08.4f}{lon:+09.4f}/" # THIS IS THE QUICKTIME FIX!!! still doesn't work for google photos though (it's also freaking out with the time when I upload from safari)
        # TODO: invesigate why sometimes this does or does not work. I have a video where the extracted mp4 doesnt have a location but the overlayed mp4 (which copies very explicitly from the extracted mp4 does (with apple photos, google photos is still a weirdo)*shrug*)
        # on the positive side at least the jpegs aren't affected by this keys stuff. Seems that adding the userdata and quicktime entires fixed the location consistency

    }

    # Apply Metadata using pyexiftool
    base_file_path = os.path.basename(file_path)
    try:
        with ExifToolHelper() as et:
            # The execute method is used for writing. It handles escaping and execution.
            # -overwrite_original tells ExifTool to directly modify the file.
            et.execute(
                "-overwrite_original",
                # Map Python dictionary keys/values to ExifTool -TAG=VALUE format
                *[f"-{k}={v}" for k, v in metadata_tags.items()],
                file_path,
                "-m"
            )

        logger.debug(f"Metadata updated successfully using pyexiftool: '{base_file_path}'")

    except FileNotFoundError:
        logger.error(f"Error: The external **ExifTool utility was not found**.")
        logger.error("Please ensure ExifTool is installed on your system and available in the PATH.")
    except Exception as e:
        logger.error(f"An error occurred during metadata writing for '{base_file_path}': {e}")

    # Changing date of capture to unix timestamp
    dt_object = pd.to_datetime(timestamp_str, utc=True)
    unix_timestamp = dt_object.timestamp()

    # Changing the OS-level timestamps
    try:
        # Set both access time and modification time to the capture time
        os.utime(file_path, (unix_timestamp, unix_timestamp))
        logger.debug(f"OS Filesystem timestamps updated: '{os.path.basename(file_path)}'")
    except Exception as e:
        logger.error(f"Failed to update filesystem time for {file_path}: {e}")


def _update_memories_metadata(df:pd.DataFrame):
    """Update the metadata using the time and location"""
    not_zips = df[(df["zip"]==False) & (df["metadata_updated"]==False)]
    total_updates = len(not_zips)
    completed_updates = 0

    # apparently lots of the pngs here are actually webps
    with logging_redirect_tqdm():
        with tqdm(total=total_updates, desc="Fixing pngs/webps") as pbar:
            for index, row in not_zips.iterrows():
                old_path = row['file_path']
                if old_path.endswith('.png') and is_actually_webp(old_path):
                    new_path = old_path.replace('.png', '.webp')
                    os.rename(old_path, new_path)
                    # Update the DataFrame so ExifTool knows the new name
                    not_zips.at[index, 'file_path'] = new_path #type:ignore
                pbar.update(1)

    # don't forget to preserve the original dataframe
    df.update(not_zips)

    with logging_redirect_tqdm():
        with tqdm(total=total_updates, desc="Updating metadata") as pbar:
            for i, row in not_zips.iterrows():
                try:
                    file_path = row["file_path"]
                    timestamp_str = row["epoch_str"]
                    lat = row["lat"]
                    lon = row["long"]
                    base_file_path = os.path.basename(file_path)

                    # Updating media
                    _update_media_metadata_pyexiftool(file_path, timestamp_str, lat, lon)

                    completed_updates += 1
                    logger.info(f"[{completed_updates}/{total_updates}] Successfully updated: '{base_file_path}'")
                    # TODO: update the dataframe would be nice here
                    not_zips.at[i, "metadata_updated"] = True #type:ignore
                except:
                    logger.error(f"Error updating {file_path}, {timestamp_str}")
                finally:
                    pbar.update(1)
    df.update(not_zips)
    return df

def get_image_dims(file_path):
    """Replacement for magick identify -format %wx%h"""
    cmd = ["magick", "identify", "-format", "%wx%h", str(file_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

def _overlay_images(df:pd.DataFrame, output_dir_str: str):
    output_dir = Path(output_dir_str)
    # Find all potential overlay files (*_1.png and *_2.png)
    overlay_files = list(output_dir.glob("*_1.webp")) + list(output_dir.glob("*_2.webp")) + list(output_dir.glob("*_1.png")) + list(output_dir.glob("*_2.png")) # TODO should change this to be dataframe based so we can track progress
    
    with logging_redirect_tqdm():
        with tqdm(total=len(overlay_files), desc="Creating overlays") as pbar:
            for png_file in overlay_files:
                png_path = Path(png_file)
                # Get base name (e.g., 00011_1760752281000_image_extracted)
                base = "_".join(png_path.stem.split("_")[:-1])
                
                # Look for background files with the same base
                # Logic: find files starting with base, case-insensitive extensions, excluding itself
                bg_extensions = ('.jpg', '.jpeg', '.mpg', '.mov', '.mp4')
                bg_file = None
                
                for match in output_dir.glob(f"{base}_*"):
                    if match.suffix.lower() in bg_extensions and match.name != png_file:
                        bg_file = match
                        break
                
                if not bg_file or not bg_file.exists():
                    print(f"⚠️ Background for {png_file} not found.")
                    continue

                ext = bg_file.suffix.lower()
                #print(f"🚀 Processing: {png_file} over {bg_file}...")

                if ext in ['.jpg', '.jpeg']:
                    # --- IMAGE WORKFLOW ---
                    output_path = output_dir / f"{base}_merged.jpg"
                    dims = get_image_dims(bg_file)
                    
                    # Composite using ImageMagick
                    # The '!' in resize forces exact dimensions
                    subprocess.run([
                        "magick", str(bg_file),
                        "(", str(png_path), "-resize", f"{dims}!", ")",
                        "-composite", str(output_path)
                    ])
                    
                    # Copy Metadata via ExifTool
                    subprocess.run([
                        "exiftool", "-overwrite_original", "-tagsFromFile", str(bg_file),
                        "-d", "%Y:%m:%d %H:%M:%S%:z", "-All:All",
                        "-DateTimeOriginal<DateTimeOriginal", "-CreateDate<CreateDate",
                        "-ModifyDate<ModifyDate", "-FileModifyDate<FileModifyDate",
                        "-OffsetTime<OffsetTime", "-OffsetTimeOriginal<OffsetTimeOriginal",
                        "-OffsetTimeDigitized<OffsetTimeDigitized", str(output_path)
                    ])
                    try:
                        # Set both access time and modification time to the capture time
                        unix_timestamp = os.path.getmtime(str(bg_file))
                        os.utime(str(output_path), (unix_timestamp, unix_timestamp))
                        logger.debug(f"OS Filesystem timestamps updated: '{os.path.basename(str(output_path))}'")
                    except Exception as e:
                        logger.error(f"Failed to update filesystem time for {str(output_path)}: {e}")

                else:
                    # --- VIDEO WORKFLOW ---
                    output_path = output_dir / f"{base}_merged.mp4"
                    temp_overlay = "temp_overlay.png"
                    
                    # Create cleaned temp overlay
                    subprocess.run(["magick", str(png_path), temp_overlay])
                    
                    # FFmpeg Overlay
                    subprocess.run([
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-i", str(bg_file), "-i", temp_overlay,
                        "-filter_complex", "[1:v]scale=rw:rh[ovr];[0:v][ovr]overlay=0:0:format=auto",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "slow", 
                        "-crf", "18", "-c:a", "copy", str(output_path), "-y"
                    ])
                    
                    # Copy Metadata
                    subprocess.run([
                        "exiftool", "-overwrite_original", "-tagsFromFile", 
                        str(bg_file), "-All:All", str(output_path)
                    ])
                    # TODO write in the OS-timestamp
                    try:
                        # Set both access time and modification time to the capture time
                        unix_timestamp = os.path.getmtime(str(bg_file))
                        os.utime(str(output_path), (unix_timestamp, unix_timestamp))
                        logger.debug(f"OS Filesystem timestamps updated: '{os.path.basename(str(output_path))}'")
                    except Exception as e:
                        logger.error(f"Failed to update filesystem time for {str(output_path)}: {e}")
                    if os.path.exists(temp_overlay):
                        os.remove(temp_overlay)
                pbar.update(1)