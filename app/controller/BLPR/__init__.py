import torch
import sys
from app.extensions import config


def get_sticker_detector():
    if config.STICKER_DETECTION_VERSION == "yolov5":
        from app.vision.detection.yolov5.detection import Yolov5Detector
        return Yolov5Detector(model_path=config.STICKER_DETECTION_YOLOV5_MODEL_PATH,
                              imgz=config.STICKER_DETECTION_YOLOV5_IMAGE_SIZE,
                              n_class=config.STICKER_DETECTION_YOLOV5_N_CLASS,
                              conf_thres=config.STICKER_DETECTION_YOLOV5_CONF_THRES,
                              iou_thres=config.STICKER_DETECTION_YOLOV5_IOU_THRES,
                              device=config.STICKER_DETECTION_DEVICE,
                              stride=config.STICKER_DETECTION_YOLOV5_STRIDE)
    else:
        raise Exception(
            "unsupported Plate detector version: {}".format(config.STICKER_DETECTION_VERSION))


def get_text_detector_landmark():
    if config.TEXT_DETECTION_VERSION == "yolov5lm":
        sys.path.insert(0, './app/vision/detection/yolov5_landmark')
        from app.vision.detection.yolov5_landmark.detection import Yolov5LMDetector
        return Yolov5LMDetector(model_path=config.TEXT_DETECTION_YOLOV5LM_MODEL_PATH,
                                img_size=config.TEXT_DETECTION_YOLOV5LM_IMAGE_SIZE,
                                conf_thres=config.TEXT_DETECTION_YOLOV5LM_CONF_THRES,
                                iou_thres=config.TEXT_DETECTION_YOLOV5LM_IOU_THRES)
    else:
        raise Exception("unsupported BLPR detector version: {}".format(config.TEXT_DETECTION_VERSION))


def get_text_detector_paddle():
    if config.TXT_DETECTION_VERSION == 'paddle':
        from app.vision.detection.paddle.detection import PaddleOcrDetector
        return PaddleOcrDetector(model_path=config.TXT_DETECTION_MODEL_PATH)
    else:
        raise Exception("unsupported BLPR Text detector version: {}".format(config.TEXT_DETECTION_VERSION))


def get_postprocessor():
    if config.POSTPROCESS_VERSION == 'format':
        from app.vision.postprocess.format.postprocess import FormatPostprocessor
        return FormatPostprocessor()
    elif config.POSTPROCESS_VERSION == 'nop':
        from app.vision.postprocess.nop.postprocess import NOPPostprocessor
        return NOPPostprocessor()
    else:
        raise Exception("unsupported BLPR Postprocessor version: {}".format(config.POSTPROCESS_VERSION))


def get_text_recognizer():
    # if config.TEXT_RECOGNITION_VERSION == "vietocr":
    #    from app.vision.recognition.vietocr.recognition import VietOCRRecognizer
    #     return VietOCRRecognizer(config=config.TEXT_RECOGNITION_VIETOCR_CONFIG,
    #                              model_path=config.TEXT_RECOGNITION_VIETOCR_MODEL_PATH)
    if config.TEXT_DETECTION_VERSION == "paddle":
        from app.vision.recognition.paddle.recognition import PaddleOcrRecognizer
        return PaddleOcrRecognizer(model_path=config.LP_RECOGNITION_PADDLE_MODEL_PATH)
    elif config.TEXT_RECOGNITION_VERSION == "tocr":
        from app.vision.recognition.ocr.recognition import TextRecognizer
        return TextRecognizer(model_path=config.TEXT_RECOGNITION_TOCR_MODEL_PATH)
    else:
        raise Exception("unsupported BLPR Text recognizer version: {}".format(config.TEXT_RECOGNITION_VERSION))


def get_algorithm():
    from app.controller.BLPR.algorithm import Algorithm
    with torch.no_grad():
        # sticker_detector = get_sticker_detector()
        text_detector_landmark = get_text_detector_landmark()
        text_detection = get_text_detector_paddle()
        text_recognizer = get_text_recognizer()
        #postprocessor = get_postprocessor()
    # return Algorithm(sticker_detector, text_detector, text_recognizer, postprocessor)
    return Algorithm(text_detector_landmark, text_detection, text_recognizer )
