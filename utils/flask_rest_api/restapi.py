"""
Run a rest API exposing the yolov5s object detection model
"""
import os
import urllib
from flask import Flask, request
import subprocess
from flask import Response
import json
from urllib.parse import urlparse

app = Flask(__name__)
DETECT = "/detect"
# DETECTION_URL = "/v1/object-detection/yolov5s"


# @app.route(DETECTION_URL, methods=["POST"])
# def predict():
#     if not request.method == "POST":
#         return

#     if request.files.get("image"):
#         image_file = request.files["image"]
#         image_bytes = image_file.read()

#         img = Image.open(io.BytesIO(image_bytes))

#         results = model(img, size=640)  # reduce size=320 for faster inference
#         return results.pandas().xyxy[0].to_json(orient="records")

@app.route(DETECT, methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        file = urlparse(request.json['filepath'])
        fileName = urllib.parse.quote(os.path.basename(file.path), safe='')
        # subprocess.run("ls")
        subprocess.run(['python3', '../../detect.py', 
                '--source', 
                request.json['filepath'], '--weights', '../../runs/train/yolo_road_det4/weights/best.pt', "--conf", "0.25", "--save-txt", "--save-conf"])
        line = ''
        with open('image_analysis.txt', 'r') as analysisFile:
            for line_from_file in analysisFile:
                line += line_from_file
        return Response(
            response=json.dumps({
                "data": {
                    "url": f'https://k-pics.s3.ap-south-1.amazonaws.com/{fileName}',
                    "line": line
                }
            }),
            status=201,
            mimetype="application/json"
        )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Flask API exposing YOLOv5 model")
    # parser.add_argument("--port", default=5000, type=int, help="port number")
    # args = parser.parse_args()

    # model = torch.hub.load("ultralytics/yolov5", "yolov5s", force_reload=True)  # force_reload to recache
    app.run(host="0.0.0.0")  # debug=True causes Restarting with stat
