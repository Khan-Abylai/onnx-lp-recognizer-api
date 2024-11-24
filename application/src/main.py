import time
import numpy as np
import warnings
import logging
from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .utils import readb64
from .detection_service import LicensePlateDetector
from .recognizer_service import LicensePlateRecognizer
from .config import settings


warnings.filterwarnings("ignore")
np.seterr(all="ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M:%S')
app = FastAPI(
    title=settings.APP_NAME,
    description="API for license plate detection and recognition",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = LicensePlateDetector(settings.MODEL_PATH_DETECTOR, image_width=settings.IMAGE_WIDTH, image_height=settings.IMAGE_HEIGHT)
recognizer = LicensePlateRecognizer(settings.MODEL_PATH_RECOGNIZER)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post('/api/checkImage')
async def analyze_route(
    file: UploadFile = File(...),
    request: Request = None
):
    """
    Analyze an image for license plate detection and recognition.
    """
    try:
        # Read file content
        contents = await file.read()
        file_size = len(contents)
        
        # Validate file size
        if file_size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400, 
                detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE/1024/1024:.1f}MB"
            )

        # Validate file type
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400, 
                detail=f"File type not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )

        # Process image
        image = readb64(contents)

        t1 = time.time()
        
        # Detection
        result, plate_img, flag = detector.run(image)
        
        if not flag:
            return JSONResponse(
                status_code=404,
                content={"status": False, "data": "No license plate detected"}
            )

        # Recognition
        label, prob = recognizer.run(plate_img)
        
        t2 = time.time()
        exec_time = t2 - t1

        return JSONResponse(content={
            'status': True,
            'data': {
                'label': label,
                'confidence': float(prob),
                'exec_time': exec_time
            }
        })

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }