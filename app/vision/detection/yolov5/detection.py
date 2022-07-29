import numpy as np
import torch
import torch.nn as nn


from app.vision.detection.yolov5.constant import yolo_config
from app.vision.detection.yolov5.models.common import Conv
from app.vision.detection.yolov5.models.experimental import attempt_load
from app.vision.detection.yolov5.models.yolo import Detect, Model
from app.vision.detection.yolov5.utils.augmentations import letterbox
from app.vision.detection.yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from app.common.base import DetectionBase


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


class Yolov5Detector(DetectionBase):
    def __init__(self, model_path, imgz, conf_thres, n_class, iou_thres,
                 device, stride=64, max_det=1000):
        super().__init__("barcode", "yolov5")
        if device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.imgz = int(imgz)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.stride = stride
        self.max_det = max_det
        self.n_class = int(n_class)
        cfg = yolo_config["yolov5s"]
        self.model = Model(cfg, nc=self.n_class).to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model = self.model.float().fuse().eval()
        for m in self.model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    def postprocess(self, results):
        coord = []
        results = (results).detach().cpu().numpy()
        #print(results[:, :4])
        for result in results:
            # x y w h
            coord.append([int(result[0]), int(result[1]), int(result[2] - result[0]), int(result[3] - result[1]), result[4]])
        return coord

    def _detect(self, img0):
        img0 = img0[0]
        imgsz = check_img_size(imgsz=self.imgz, s=self.stride)

        # Padded resize
        img = letterbox(img0, (imgsz, imgsz), stride=self.stride, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = self.model(img, augment=False, visualize=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=None, max_det=self.max_det)
        results = []
        for _, det in enumerate(pred):  # per image
            if len(det):
                # print(det)
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                results.append({'image': img0,
                                'coordinates': det[:, :4].detach().cpu().numpy().astype(int).tolist()})
                print(results)
        return results