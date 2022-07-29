from app.common.base import DetectionBase, RecognitionBase, AlgorithmBase, PostprocessorBase, DetectionResult, \
    RecognizerBasePaddle
from app.common.model import BarcodeResults, DetectionResult2
from app.vision.utils import preprocess_crop_rec, preprocess_crop_affine, preprocess_draw_rec, draw_landmark, \
    preprocess_draw_yolo
import time


class Algorithm(AlgorithmBase):
    def __init__(self,
                 # sticker_detector: DetectionBase):
                 text_detector_landmark: DetectionBase,
                 txt_detection_paddle: DetectionBase,
                 text_recognizer: RecognitionBase):
        # postprocessor: PostprocessorBase):
        super().__init__()
        # self._sticker_detector = sticker_detector
        self._text_detector_landmark = text_detector_landmark
        self._txt_detection_paddle = txt_detection_paddle
        self._text_recognizer = text_recognizer
        # self._postprocessor = postprocessor

    def process(self, org_img):
        # sticker_results = self._sticker_detector.detect([org_img])
        # if len(sticker_results.results) == 0:
        #    return BarcodeResults(org_img=org_img, barcodes=[])
        # crop and draw bbox image with yolov5
        # sticker_images = preprocess_crop_rec(sticker_results.results)
        # sticker_images = preprocess_draw_rec(sticker_results.results)
        start_time = time.time()
        text_detection_results = self._text_detector_landmark.detect([org_img])
        end_time = time.time()
        time_taken = end_time - start_time
        print(f'Time taken: {time_taken}')
        if len(text_detection_results.results) == 0:
            return BarcodeResults(org_img=org_img, barcodes=[])
        # crop and draw bbox with yolo-landmark
        # text_images = preprocess_draw_rec(text_detection_results.results)
        text_images = preprocess_crop_affine(text_detection_results.results)
        text_detection = self._txt_detection_paddle.detect(text_images[0])
        print('ABC')
        if len(text_detection.results) == 0:
            return BarcodeResults(org_img=org_img, barcodes=[])
        line_text_detection = preprocess_draw_yolo(text_detection.results)
        # postprocessed_results = self._postprocessor.postprocess(text_values.results)
        return DetectionResult2(org_img=org_img, barcodes=line_text_detection)
        # return BarcodeResults(org_img=org_img, barcodes=postprocessed_results)
