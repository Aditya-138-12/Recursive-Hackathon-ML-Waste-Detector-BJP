from flask import Flask, request, jsonify, make_response
import os
import cv2
from image_processor import process_image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST")
    return response

def _build_cors_headers():
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST"
    }
    return headers

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == "OPTIONS":  # CORS preflight
        return _build_cors_preflight_response()

    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        # Process the uploaded file using your image processing logic
        model_path = "/home/aditya/Desktop/data/runs/detect/train/weights/last.pt"
        results = process_image(file_path, model_path)

        # Assuming you convert processed image to PNG and encode to bytes
        _, img_encoded = cv2.imencode('.png', results["processed_image"])
        img_bytes = img_encoded.tobytes()

        # Prepare JSON response
        response_data = {
            "area": results["area"],
            "image_size": results["image_size"],
            "percentage": results["percentage"],
            "latitude": results["latitude"],
            "longitude": results["longitude"],
            "filePath": file_path,
            "processed_image": img_bytes.hex()
             # Convert bytes to hex for JSON
        }

        return jsonify(response_data), 200, _build_cors_headers()

    # Handle other methods if needed
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    app.run(debug=True)

