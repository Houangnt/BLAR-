from abc import ABCMeta, abstractmethod
from typing import Tuple, List
from app.common.model import DetectionResult, DetectionResults, RecognitionResults, RecognitionResultPaddle


class DetectionBase(metaclass=ABCMeta):
    def __init__(self, detection_type: str, version: str):
        self._detection_type = detection_type
        self._version = version

    def detect(self, imgs) -> DetectionResults:
        results = self._detect(imgs)
        results = [DetectionResult.from_dict(result) for result in results]
        return DetectionResults(results=results, version=self._version)

    @abstractmethod
    def _detect(self, imgs):
        pass


class RecognitionBase(metaclass=ABCMeta):
    def __init__(self, recognition_type: str, version: str):
        self._recogniton_type = recognition_type
        self._version = version

    def recognize(self, imgs):
        results = self._recognize(imgs)
        return RecognitionResults(results=results, version=self._version)

    @abstractmethod
    def _recognize(self, imgs):
        pass


class RecognizerBasePaddle(metaclass=ABCMeta):
    def __init__(self, version: str, engine=None):
        self._engine = engine
        self._version = version

    @abstractmethod
    def _recognize(self, box) -> Tuple[str, float]:
        pass

    def get_version(self):
        return self._version

    def recognize(self, box) -> RecognitionResultPaddle:
        value, score = self._recognize(box)
        return RecognitionResultPaddle(
            value=value,
            score=score,
            version=self._version,
            engine=self._engine,
        )


class AlgorithmBase(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def process(self, img):
        pass


class PostprocessorBase(metaclass=ABCMeta):
    @abstractmethod
    def _postprocess(self, results):
        pass

    def postprocess(self, result):
        postprocessed_result = self._postprocess(result)
        return postprocessed_result
