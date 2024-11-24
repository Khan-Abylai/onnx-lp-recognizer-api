from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "License Plate Recognition API"
    DEBUG: bool = False
    MODEL_PATH_DETECTOR: str = "models/detector_model_retinaface.onnx"
    MODEL_PATH_RECOGNIZER: str = "models/recognizer_base.onnx"
    IMAGE_WIDTH: int = 640
    IMAGE_HEIGHT: int = 480
    MAX_FILE_SIZE: int = 5 * 1024 * 1024  # 5MB
    ALLOWED_EXTENSIONS: set = {"jpg", "jpeg", "png"}
    
    class Config:
        env_file = ".env"

settings = Settings()

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 16,
    'ngpu': 1,
    'epoch': 50,
    'decay1': 190,
    'decay2': 220,
    'image_size': 480,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 10,
    'decay1': 70,
    'decay2': 90,
    'image_size': 480,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

