import logging
import os
import pathlib
import sys

from dotenv import load_dotenv

env = load_dotenv()

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'app.log'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

FORMATTER = logging.Formatter(
    "%(asctime)s — %(filename)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")


def get_console_handler():
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_logger(*, logger_name):
    """Get logger with prepared handlers."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(FORMATTER)

    logger.addHandler(file_handler)
    logger.addHandler(get_console_handler())
    logger.propagate = False

    return logger


logger = get_logger(logger_name=__name__)


class ApplicationConfig:
    PLATE_DETECTION_VERSION=os.environ.get('PLATE_DETECTION_VERSION')
    PLATE_DETECTION_DEVICE=os.environ.get('PLATE_DETECTION_DEVICE')
    PLATE_DETECTION_YOLOV5_MODEL_PATH=os.environ.get('PLATE_DETECTION_YOLOV5_MODEL_PATH')
    PLATE_DETECTION_YOLOV5_IMAGE_SIZE=int(os.environ.get('PLATE_DETECTION_YOLOV5_IMAGE_SIZE'))
    PLATE_DETECTION_YOLOV5_N_CLASS=int(os.environ.get('PLATE_DETECTION_YOLOV5_N_CLASS'))
    PLATE_DETECTION_YOLOV5_CONF_THRES=float(os.environ.get('PLATE_DETECTION_YOLOV5_CONF_THRES'))
    PLATE_DETECTION_YOLOV5_IOU_THRES=float(os.environ.get('PLATE_DETECTION_YOLOV5_IOU_THRES'))
    PLATE_DETECTION_YOLOV5_STRIDE=int(os.environ.get('PLATE_DETECTION_YOLOV5_STRIDE'))

    TEXT_DETECTION_VERSION=os.environ.get('TEXT_DETECTION_VERSION')
    TEXT_DETECTION_DEVICE=os.environ.get('TEXT_DETECTION_DEVICE')
    TEXT_DETECTION_YOLOV5LM_MODEL_PATH=os.environ.get('TEXT_DETECTION_YOLOV5LM_MODEL_PATH')
    TEXT_DETECTION_YOLOV5LM_CONF_THRES=float(os.environ.get('TEXT_DETECTION_YOLOV5LM_CONF_THRES'))
    TEXT_DETECTION_YOLOV5LM_IOU_THRES=float(os.environ.get('TEXT_DETECTION_YOLOV5LM_IOU_THRES'))
    TEXT_DETECTION_YOLOV5LM_IMAGE_SIZE=int(os.environ.get('TEXT_DETECTION_YOLOV5LM_IMAGE_SIZE'))

    TEXT_RECOGNITION_VERSION=os.environ.get('TEXT_RECOGNITION_VERSION')
    TEXT_RECOGNITION_DEVICE=os.environ.get('TEXT_RECOGNITION_DEVICE')
    BLPR_RECOGNITION_PADDLE_MODEL_PATH = os.environ.get('BLPR_RECOGNITION_PADDLE_MODEL_PATH')
    TEXT_RECOGNITION_VIETOCR_MODEL_PATH=os.environ.get('TEXT_RECOGNITION_VIETOCR_MODEL_PATH')
    TEXT_RECOGNITION_VIETOCR_CONFIG=os.environ.get('TEXT_RECOGNITION_VIETOCR_CONFIG')
    TEXT_RECOGNITION_TOCR_MODEL_PATH = os.environ.get('TEXT_RECOGNITION_TOCR_MODEL_PATH')
    BLPR_RECOGNITION_PADDLE_IMG_SIZE = os.environ.get('BLPR_RECOGNITION_PADDLE_IMG_SIZE')
    BLPR_RECOGNITION_VOCAB = os.environ.get('BLPR_RECOGNITION_VOCAB')

    TXT_DETECTION_VERSION = os.environ.get("TXT_DETECTION_VERSION")
    TXT_DETECTION_MODEL_PATH = os.environ.get("TXT_DETECTION_MODEL_PATH")

    DOWNLOAD_IMG_RETRY=int(os.environ.get('DOWNLOAD_IMG_RETRY'))
    GS_TIMEOUT=int(os.environ.get('GS_TIMEOUT'))
    RETRY_TIME=float(os.environ.get('RETRY_TIME'))

    NUM_OF_WORKERS=int(os.environ.get('NUM_OF_WORKERS'))
    INPUT_DIR=os.environ.get('INPUT_DIR')
    DUMP_DIR=os.environ.get('DUMP_DIR')

    POSTPROCESS_VERSION = os.environ.get('POSTPROCESS_VERSION')

config = ApplicationConfig()
