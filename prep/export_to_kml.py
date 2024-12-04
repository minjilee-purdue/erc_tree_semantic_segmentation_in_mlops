import os
import numpy as np
from sklearn.cluster import DBSCAN
from simplekml import Kml, Style

def parse_gps_files(folder_path):
    """Parse all .txt files to extract GPS coordinates."""
    gps_coordinates = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt") and not file_name.startswith("._"):  # Skip macOS hidden files
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as f:
                try:
                    lines = f.readlines()
                    
                    # Extract Latitude and Longitude from respective lines
                    latitude_line = [line for line in lines if "GPS Latitude" in line][0]
                    longitude_line = [line for line in lines if "GPS Longitude" in line][0]

                    latitude = latitude_line.split("\t")[1].strip()  # Extract value after the tab
                    longitude = longitude_line.split("\t")[1].strip()

                    # Convert DMS (degrees, minutes, seconds) to Decimal Degrees
                    latitude_decimal = dms_to_decimal(latitude[:-1])  # Remove "N"
                    longitude_decimal = dms_to_decimal(longitude[:-1])  # Remove "W"
                    longitude_decimal = -longitude_decimal  # Negate for Western Hemisphere
                    
                    gps_coordinates.append([latitude_decimal, longitude_decimal])
                except (IndexError, ValueError) as e:
                    print(f"Skipping file {file_name} due to error: {e}")
    return np.array(gps_coordinates)

def dms_to_decimal(dms):
    """Convert DMS (degrees, minutes, seconds) string to decimal degrees."""
    degrees, minutes, seconds = map(float, dms.replace("º", " ").replace("'", " ").replace('"', " ").split())
    return degrees + (minutes / 60) + (seconds / 3600)

def export_to_kml(coordinates, clusters, output_file):
    """Export GPS points and DBSCAN clusters to a KML file."""
    kml = Kml()

    # Define styles for clusters
    cluster_styles = {}
    for cluster in np.unique(clusters):
        if cluster == -1:
            style = Style()
            style.iconstyle.color = 'ff0000ff'  # Red for noise
            style.iconstyle.scale = 1
        else:
            style = Style()
            style.iconstyle.color = 'ff00ff00'  # Green for clusters
            style.iconstyle.scale = 1
        cluster_styles[cluster] = style

    # Add points to the KML file
    for i, (lat, lon) in enumerate(coordinates):
        cluster = clusters[i]
        pnt = kml.newpoint(coords=[(lon, lat)])
        pnt.style = cluster_styles[cluster]
        if cluster == -1:
            pnt.name = f"Noise Point {i+1}"
        else:
            pnt.name = f"Cluster {cluster} - Point {i+1}"

    # Save the KML file
    kml.save(output_file)
    print(f"KML file saved to {output_file}")

if __name__ == "__main__":
    gps_folder = "/media/minjilee/Backup Plus/seasonaldata/feb/feb_jpg_original/feb_data/bg/extracted"
    
    # Parse GPS files
    gps_data = parse_gps_files(gps_folder)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.01, min_samples=5, metric='haversine')
    clusters = dbscan.fit_predict(np.radians(gps_data))
    
    # Export results to a KML file
    kml_output_file = "/home/minjilee/Desktop/dbscan_clusters.kml"
    export_to_kml(gps_data, clusters, kml_output_file)
