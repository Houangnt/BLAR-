from typing import List
from dataclasses import dataclass
from dataclasses_json import dataclass_json, LetterCase
import numpy as np


@dataclass_json(letter_case=LetterCase.SNAKE)
@dataclass
class DetectionResult:
    image: np.array
    coordinates: List[List]


@dataclass_json(letter_case=LetterCase.SNAKE)
@dataclass
class DetectionResults:
    version: str
    results: List[DetectionResult]


@dataclass_json(letter_case=LetterCase.SNAKE)
@dataclass
class RecognitionResults:
    version: str
    results: List[str]


@dataclass_json(letter_case=LetterCase.SNAKE)
@dataclass
class BLARResults:
    org_img: np.array
    plate: List[str]


@dataclass_json(letter_case=LetterCase.SNAKE)
@dataclass
class DetectionResult2:
    org_img: np.array
    plate: np.array


# Paddle
@dataclass
class TextDetection:
    coordinate: List
    version: str


@dataclass_json(letter_case=LetterCase.SNAKE)
@dataclass
class RecognitionResultPaddle:
    engine: str
    score: float
    value: str
    version: str
