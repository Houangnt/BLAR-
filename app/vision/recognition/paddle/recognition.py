from typing import Tuple

from app.vision.paddleocr_libs.predict_rec import PaddleOCRTextRecognizer

from app.common.base import RecognizerBasePaddle

from app.extensions import config
from os.path import splitext


class PaddleOcrRecognizer(RecognizerBasePaddle):
    def __init__(self, model_path):
        super().__init__(version="paddle", engine="alpr-vision")
        self.rec_char_dict_path = splitext(model_path)[0] + '.txt'
        self.model = PaddleOCRTextRecognizer(lang="en",
                                             rec_model_dir=model_path,
                                             rec_char_type="ch",
                                             rec_image_shape=config.BLPR_RECOGNITION_PADDLE_IMG_SIZE,
                                             rec_char_dict_path=self.rec_char_dict_path,
                                             use_gpu=False)

    def _recognize(self, box) -> Tuple[str, float]:
        if not isinstance(box, list):
            box = [box]
        results = self.model(box)
        value, score = results[0]
        return value, score
