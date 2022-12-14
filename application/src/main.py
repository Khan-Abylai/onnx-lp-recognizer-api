import time
import numpy as np
import warnings
import logging
from fastapi import FastAPI
from fastapi import Request
from utils import readb64
from detection_service import LicensePlateDetector
from recognizer_service import LicensePlateRecognizer

warnings.filterwarnings("ignore")
np.seterr(all="ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
app = FastAPI()
detector = LicensePlateDetector('detector_model_retinaface_trained.onnx', image_width=640, image_height=480)
recognizer = LicensePlateRecognizer('recognizer_base.onnx')


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post('/api/checkImage')
async def analyze_route(request: Request):
    form = await request.form()
    if "image" in form:
        t1 = time.time()
        upload_file = form["image"]
        filename = form["image"].filename  # str
        image_base64 = await form["image"].read()  # bytes
        content_type = form["image"].content_type  # str
        image = readb64(image_base64)
        t2 = time.time()

        print(f"Time for reading image:{t2 - t1} seconds")

        result, plate_img, flag = detector.run(image)
        if flag:
            label, prob = recognizer.run(plate_img)
            return {'status': True, 'data': {'label': label, 'prob': prob}}
        else:
            return {"status": False, "data": 'Not Recognized'}
    else:
        return {"status": False, "data": None}


@app.post("/api/image")
async def analyze_route(request: Request):
    form = await request.form()
    if "image" in form:
        t1 = time.time()
        upload_file = form["image"]
        filename = form["image"].filename  # str
        image_base64 = await form["image"].read()  # bytes
        content_type = form["image"].content_type  # str
        image = readb64(image_base64)
