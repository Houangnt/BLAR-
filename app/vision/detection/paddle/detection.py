from app.vision.paddleocr_libs.predict_det import PaddleOCRTextDetector


from app.common.base import DetectionBase
from app.common.model import TextDetection

import numpy as np


class PaddleOcrDetector(DetectionBase):
    def __init__(self, model_path):
        super().__init__(detection_type='alpr', version="paddle")
        self.model = PaddleOCRTextDetector(det_model_dir=model_path,
                                           use_gpu=False,
                                           lang='en')

    def _detect(self, img):
        img0 = img[0]
        txt_detection_results = []
        four_pts_results = self.model(img)
        xywh_results = np.empty((0, 4), int)
        for i in range(four_pts_results.shape[0]):
            xywh_results = np.append(xywh_results,
                                     self._convert_to_xywh(four_pts_results[i, :, :]),
                                     axis=0)
        xywh_results = np.delete(xywh_results,
                                 np.where((xywh_results[:, 2] < 50) | (xywh_results[:, 3] < 50))[0],
                                 axis=0)
        xywh_results = xywh_results[xywh_results[:, 1].argsort()]
        if xywh_results.shape[0] > 1:
            txt_list = self._merge_boxes(xywh_results)
        else:
            txt_list = xywh_results.tolist()
        results = []
        print(f'txt paddle : {txt_list}')
        for txt in txt_list:
            print(f'txt {txt}')
            # txt_detection_results.append(TextDetection(coordinate=txt, version=self._version))
            results.append({'image' : img0,
                            'coordinates' : TextDetection(coordinate=txt, version=self._version).coordinate })
        #print(results)
        return results

    def _convert_to_xywh(self, box):
        x_min, y_min = np.amin(box, axis=0)
        x_max, y_max = np.amax(box, axis=0)
        return np.array([[int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]])

    def _intersection_segment(self, box_1, box_2):
        intersection_y = min(box_1[1], box_2[1]) - max(box_1[1] + box_1[3], box_2[1] + box_2[3])
        return max(0, 1.0 * intersection_y / (box_1[3] + box_2[3] - intersection_y))

    def _merge_boxes(self, boxes):
        upper_line = np.empty((0, 4), int)
        lower_line = np.empty((0, 4), int)
        upper_line = np.append(upper_line, np.expand_dims(boxes[0, :], axis=0), axis=0)
        for i in range(1, boxes.shape[0]):
            if self._intersection_segment(boxes[0, :], boxes[i, :]) > 0.3:
                upper_line = np.append(upper_line, np.expand_dims(boxes[i, :], axis=0), axis=0)
            else:
                lower_line = np.append(lower_line, np.expand_dims(boxes[i, :], axis=0), axis=0)
        text_list = [self._merge(upper_line), self._merge(lower_line)]
        return text_list

    def _merge(self, boxes):
        if boxes.shape[0] > 1:
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            x_min, y_min = np.amin(boxes[:, :2], axis=0)
            x_max, y_max = np.amax(boxes[:, 2:], axis=0)
            return [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
        else:
            return boxes[0, :].tolist()
