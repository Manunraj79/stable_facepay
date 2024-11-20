import os
import cv2
import json
import base64
import numpy as np
from PIL import Image, ImageFile, ImageOps
from io import BytesIO
from datetime import datetime
from demo import handler_server_today
from waitress import serve
from flask import Flask, request
ImageFile.LOAD_TRUNCATED_IMAGES = True

app = Flask(__name__)

def array_to_bytes(x):
    np_bytes = BytesIO()
    np.save(np_bytes, np.array(x), allow_pickle=True)
    return np_bytes.getvalue()

@app.route("/", methods=["GET", "POST"])
def request_api():
    now = datetime.now()
    date = now.strftime("%m:%d:%Y_%H:%M:%S.%f")
    data = json.loads(request.data)
    mode = data["mode"]
    if mode == 'train':
        result = handler_server_today.write_face_features(data)
        return result
    elif mode == 'test':
        file_print = data["img_path"]
        img1_ = base64.b64decode(file_print)
        img1_ = np.array(ImageOps.exif_transpose(Image.open(BytesIO(img1_))))
        img1_ = cv2.cvtColor(img1_, cv2.COLOR_BGR2RGB)
        result = handler_server_today.validate_user(date, img1_)
        return result

if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=8000)
