import folium
from folium import plugins
import numpy as np
import pandas as pd
import requests
import PIL.Image
import PIL.ExifTags
import base64
from io import BytesIO
from PIL.ExifTags import TAGS, GPSTAGS
import os
from ultralytics import YOLO
import torch
import cv2
import supervision as sv
from pathlib import Path

def generate_map(coords_list, output_file='map.html'):

    # Check if the list is not empty
    if not coords_list:
        raise ValueError("The coordinates list is empty.")
    
    # Create a map centered at the first coordinate in the list
    initial_location = coords_list[0]
    map_ = folium.Map(location=[initial_location[0], initial_location[1]], zoom_start=13)
    
    # Add a marker for each pair of coordinates in the list
    for lat, lon in coords_list:
        folium.Marker([lat, lon], popup=f"Marker at ({lat}, {lon})").add_to(map_)
    
    # Save the map to an HTML file
    map_.save(output_file)

    print(f"Map has been saved to {output_file}")
def generate_map_with_image_urls(coords_list, image_urls, output_file='map.html'):
    """
    Generates an HTML file with a map and places markers at each location specified in the coords_list.
    Each marker displays an image from the given URL in the popup when clicked.

    :param coords_list: List of tuples, where each tuple contains (latitude, longitude)
    :param image_urls: List of URLs corresponding to the images
    :param output_file: Name of the output HTML file (default is 'map.html')
    """
    if not coords_list or not image_urls or len(coords_list) != len(image_urls):
        raise ValueError("Coordinates list and image URLs list must be non-empty and of the same length.")
    
    # Create a map centered at the first coordinate in the list
    initial_location = coords_list[0]
    map_ = folium.Map(location=[initial_location[0], initial_location[1]], zoom_start=13)
    
    for (lat, lon), img_url in zip(coords_list, image_urls):
        # Create an HTML image tag for the popup
        img_html = f'<img src="{img_url}" width="150px"/>'
        # Create a popup with the image
        popup = folium.Popup(folium.Html(img_html, script=True), max_width=250)
        
        # Add a marker with the popup
        folium.Marker([lat, lon], popup=popup).add_to(map_)
    
    # Save the map to an HTML file
    map_.save(output_file)

    print(f"Map has been saved to {output_file}")
def generate_map_with_images(coords_list, image_list, output_file='map.html'):

    if not coords_list or not image_list or len(coords_list) != len(image_list):
        raise ValueError("Coordinates list and image list must be non-empty and of the same length.")
    
    # Create a map centered at the first coordinate in the list
    initial_location = coords_list[0]
    map_ = folium.Map(location=[initial_location[0], initial_location[1]], zoom_start=13)
    
    for (lat, lon), img in zip(coords_list, image_list):
        # Convert the PIL image to a base64 string
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        img_html = f'<img src="data:image/png;base64,{img_str}" width="150px"/>'
        
        # Create a popup with the image
        popup = folium.Popup(folium.Html(img_html, script=True), max_width=250)
        
        # Add a marker with the popup
        folium.Marker([lat, lon], popup=popup).add_to(map_)
    
    # Save the map to an HTML file
    map_.save(output_file)

    print(f"Map has been saved to {output_file}")
    
def get_exif_data(image):
    """Extract EXIF data from an image."""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            tag_name = TAGS.get(tag, tag)
            exif_data[tag_name] = value
    return exif_data

def get_gps_data(exif_data):
    """Extract GPS data from EXIF data."""
    gps_info = {}
    for key, value in exif_data.items():
        if key == "GPSInfo":
            for t in value:
                sub_decoded = GPSTAGS.get(t, t)
                gps_info[sub_decoded] = value[t]
    return gps_info


def convert_to_degrees(value):
    """Convert the GPS coordinates stored in the EXIF to degrees in float format."""
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])

    return d + (m / 60.0) + (s / 3600.0)

def extract_gps(image):
    """Extract GPS coordinates from an image."""
    exif_data = get_exif_data(image)
    gps_info = get_gps_data(exif_data)

    if not gps_info:
        return None, None

    lat = convert_to_degrees(gps_info['GPSLatitude'])
    lon = convert_to_degrees(gps_info['GPSLongitude'])

    # Adjust for N/S and E/W
    if gps_info['GPSLatitudeRef'] == 'S':
        lat = -lat
    if gps_info['GPSLongitudeRef'] == 'W':
        lon = -lon

    return lat, lon
box_annot = sv.BoxAnnotator(thickness=3, text_thickness=3, text_scale=5)
deploy_Model = YOLO("D:/PYTHON/Manhole Cover Detection/map/AI_Models/Label_2_2.pt")
Folder = "D:/PYTHON/Manhole Cover Detection/map/GPS MANHOLE PHOTOS/"
pathOBJ = Path(Folder)
file_paths = [str(file) for file in pathOBJ.rglob('*') if file.is_file()]
lat_lons = []
images = []
j=0
directory = os.path.dirname("map_images/")
if not os.path.exists(directory):os.makedirs(directory)
for i in file_paths:
    img_frame = cv2.imread(i)
    result = deploy_Model(img_frame)[0]
    detection = sv.Detections.from_yolov8(result)
    labels = [
        f"{deploy_Model.names[class_id]} {confidence:0.01f}"
        for _, confidence, class_id, _
        in detection if confidence>=0.5
    ]
    if len(detection)> 0:
        img_frame = box_annot.annotate(scene=img_frame, detections=detection, labels=labels)
        gps_img = PIL.Image.open(i)
        lat_lons.append((extract_gps(gps_img)))
        k = cv2.imwrite("map_images/"+str(j) + ".jpg", img_frame)
        if not k:print("fail")
        images.append("map_images/"+str(j) + ".jpg")
        j+=1
generate_map_with_image_urls(lat_lons, images)