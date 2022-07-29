# -*- coding: utf-8 -*-
"""Helper utilities and decorators."""

import base64
import json
import os
from typing import List
import time
import cv2
import newrelic.agent
import numpy as np
import requests
import torch
from app.extensions import config, logger


@newrelic.agent.function_trace()
def download_image(*, url):
    for _ in range(int(config.DOWNLOAD_IMG_RETRY)):
        try:
            resp = requests.get(url, timeout=int(config.GS_TIMEOUT))
            if not resp.ok:
                time.sleep(float(config.RETRY_TIME))
                continue
            image = np.asarray(bytearray(resp.content), dtype='uint8')
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            return image
        except Exception:
            time.sleep(float(config.RETRY_TIME))
            continue
    raise Exception("An exception of downloading image")


def encode_image(img):
    retval, buffer = cv2.imencode(".jpg", img)
    jpg_as_text = base64.b64encode(buffer.tobytes())
    return str(jpg_as_text, 'utf-8')


def decode_image(img):
    img = bytes(img, 'utf-8')
    img_arr = np.frombuffer(base64.b64decode(img), dtype=np.uint8)
    return cv2.imdecode(img_arr, flags=cv2.IMREAD_COLOR)
