import os
import subprocess
import json

def format_gps_value(gps_value, direction, force_west=False):
    """Format GPS value to replace 'deg' with 'º' and handle direction."""
    if gps_value != 'N/A':
        formatted_value = gps_value.replace("deg", "º").replace(" ", "")  # Remove extra spaces
        # Force longitude direction to 'W' if needed
        if force_west and "E" in formatted_value:
            # instead of "E", it is supposed to change to "W" - default and initialize issue
            formatted_value = formatted_value.replace("E", "W")
        return formatted_value
    return gps_value

def extract_gps_metadata_from_dng(file_path, exiftool_path):
    """Extract GPS metadata from a .DNG file using subprocess to call exiftool."""
    try:
        # Run ExifTool as a subprocess and capture output in JSON format
        result = subprocess.run(
            [exiftool_path, "-j", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        metadata_list = json.loads(result.stdout)
        if not metadata_list:
            return None

        metadata = metadata_list[0]  # Get metadata for the current file

        # Debug: Print full metadata
        print(f"Full Metadata for {file_path}: {metadata}")
        
        # Extract GPS data
        gps_altitude = metadata.get('GPSAltitude', 'N/A')
        gps_latitude = format_gps_value(metadata.get('GPSLatitude', 'N/A'), metadata.get('GPSLatitudeRef', ''))
        gps_longitude = format_gps_value(metadata.get('GPSLongitude', 'N/A'), metadata.get('GPSLongitudeRef', ''), force_west=True)

        return {
            "GPS Altitude": gps_altitude,
            "GPS Latitude": gps_latitude,
            "GPS Longitude": gps_longitude
        }
    except subprocess.CalledProcessError as e:
        print(f"Error running exiftool on {file_path}: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding metadata for {file_path}: {e}")
        return None

def process_dng_images(input_folder, output_folder, exiftool_path):
    """Process all .DNG files in a folder and save GPS metadata."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith('.dng'):  # Ensure case-insensitive file extension match
            file_path = os.path.join(input_folder, file_name)
            print(f"Processing: {file_path}")
            metadata = extract_gps_metadata_from_dng(file_path, exiftool_path)
            
            if metadata:
                print(f"Metadata extracted: {metadata}")
                output_file_path = os.path.join(output_folder, f"{file_name}_gps.txt")
                with open(output_file_path, "w") as f:
                    f.write(f"GPS Altitude\t{metadata['GPS Altitude']}\n")
                    f.write(f"GPS Latitude\t{metadata['GPS Latitude']}\n")
                    f.write(f"GPS Longitude\t{metadata['GPS Longitude']}\n")
            else:
                print(f"No GPS metadata found for: {file_path}")

if __name__ == "__main__":
    # Define input and output folders
    input_folder_path = "/home/minjilee/Desktop/test"
    output_folder_path = "/home/minjilee/Desktop/test/extracted"
    
    # Define the path to exiftool executable
    exiftool_path = "/home/minjilee/Desktop/exiftool/exiftool"
    
    # Process DNG files and extract metadata
    process_dng_images(input_folder_path, output_folder_path, exiftool_path)
    print(f"Metadata extraction completed. Files are saved in: {output_folder_path}")
