# image_processor.py

import cv2
from ultralytics import YOLO
import re
import os

def extract_lat_long(filename):
    pattern = re.compile(r'([\d.-]+)_([\d.-]+)_')
    match = pattern.search(filename)
    if match:
        latitude = float(match.group(1))
        longitude = float(match.group(2))
        return latitude, longitude
    else:
        raise ValueError("The filename does not contain latitude and longitude in the expected format.")

def process_image(imagePath, modelPath):
    filename = os.path.basename(imagePath)
    image = cv2.imread(imagePath)
    model = YOLO(modelPath)
    r_img = cv2.resize(image, (500, 500))
    results = model(r_img)

    area = 0

    def area_calc(x1, y1, x2, y2):
        length = abs(x1 - x2)
        width = abs(y1 - y2)
        return length * width

    for result in results:
        boxes = result.boxes
        boxes_list = boxes.data.tolist()
        
        for o in boxes_list:
            x1, y1, x2, y2, score, class_id = o
            pred_img = cv2.rectangle(r_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            area += area_calc(x1, y1, x2, y2)

    latitude, longitude = extract_lat_long(filename)

    return {
        "area": area,
        "image_size": 500 * 500,
        "percentage": ((area / (500 * 500)) * 100) * 100,
        "latitude": latitude,
        "longitude": longitude,
        "processed_image": r_img
    }

