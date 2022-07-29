import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
import numpy as np
from app.vision.detection.yolov5_landmark.models.experimental import attempt_load
from app.vision.detection.yolov5_landmark.utils.datasets import letterbox
from app.vision.detection.yolov5_landmark.utils.general import check_img_size, non_max_suppression_face, scale_coords, xyxy2xywh, scale_coords_landmarks, crop_affine
from app.vision.detection.yolov5_landmark.utils.torch_utils import time_synchronized
from app.common.base import DetectionBase
from glob import glob


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def show_results(img, xywh, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xywh[0] * w - 0.5 * xywh[2] * w)
    y1 = int(xywh[1] * h - 0.5 * xywh[3] * h)
    x2 = int(xywh[0] * w + 0.5 * xywh[2] * w)
    y2 = int(xywh[1] * h + 0.5 * xywh[3] * h)
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(4):
        point_x = int(landmarks[2 * i] * w)
        point_y = int(landmarks[2 * i + 1] * h)
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


class Yolov5LMDetector(DetectionBase):
    def __init__(self, model_path, img_size, conf_thres, iou_thres, device="cpu"):
        super().__init__("barcode", "yolov5_landmark")
        if device == "cuda":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = load_model(model_path, self.device)
        self.img_size = int(img_size)
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)

    def _detect(self, imgs):
        # imgs = self.preprocess(imgs)
        img_list = []
        for orgimg in imgs:
            img0 = copy.deepcopy(orgimg)
            h0, w0 = orgimg.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # resize image to img_size
            if r != 1:  # always resize down, only resize up if training with augmentation
                interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
                img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
            imgsz = check_img_size(self.img_size, s=self.model.stride.max())  # check img_size
            img = letterbox(img0, new_shape=imgsz)[0]
            ##########################################################################
            if img.shape[0] < 640:
                pad = 640 - img.shape[0]
                top, bottom = pad//2+pad%2, pad//2
                img = cv2.copyMakeBorder(img, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            elif img.shape[1] < 640:
                pad = 640 - img.shape[1]
                left, right = pad//2+pad%2, pad//2
                img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
            ##########################################################################
            img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
            img = torch.from_numpy(img).to(self.device)
            img = img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            img = img[np.newaxis, ...]
            img_list.append(img)

        batch = torch.cat(img_list, 0)
        pred = self.model(batch)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)

        results = []

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            orgimg = imgs[i]
            landmarkss = []
            gn = torch.tensor(orgimg.shape)[[1, 0, 1, 0]].to(self.device)  # normalization gain whwh
            gn_lks = torch.tensor(orgimg.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(self.device)  # normalization gain landmarks
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()
                for j in range(det.size()[0]):
                    xywh = (xyxy2xywh(det[j, :4].view(1, 4)) / gn).view(-1).tolist()
                    #print(xywh)
                    conf = det[j, 4].cpu().numpy()
                    class_num = det[j, 13].cpu().numpy()
                    landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
                    landmarkss.append(landmarks)
                    # orgimg = show_results(orgimg, xywh, conf, landmarks, class_num)
                results.append({'image': orgimg,
                                'coordinates': landmarkss})
            # cv2.imshow('result', orgimg)
            # cv2.waitKey(0)
        #print(results)
        return results







    # No batch
    # def _detect(self, org_img):
    #     results = []
    #     h0, w0 = org_img.shape[:2]  # orig hw
    #     img0 = copy.deepcopy(org_img)
    #
    #     r = self.img_size / max(h0, w0)  # resize image to img_size
    #     if r != 1:  # always resize down, only resize up if training with augmentation
    #         interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
    #         img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
    #     img = letterbox(img0, new_shape=self.img_size)[0]
    #
    #     # Convert
    #     img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416
    #
    #     img = torch.from_numpy(img).to(self.device)
    #     img = img.float()  # uint8 to fp16/32
    #     img /= 255.0  # 0 - 255 to 0.0 - 1.0
    #     if img.ndimension() == 3:
    #         img = img.unsqueeze(0)
    #
    #     # Inference
    #     pred = self.model(img)[0]
    #
    #     # Apply NMS
    #     det = non_max_suppression_face(pred, self.conf_thres, self.iou_thres)[0]
    #     if len(det):
    #         gn_lks = torch.tensor(org_img.shape)[[1, 0, 1, 0, 1, 0, 1, 0]].to(self.device)
    #         det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], org_img.shape).round()
    #         for j in range(det.size()[0]):
    #             # conf = det[j, 4].cpu().numpy()
    #             landmarks = (det[j, 5:13].view(1, 8) / gn_lks).view(-1).tolist()
    #             cropped_plate, lp_type, points = crop_affine_fixed(org_img, landmarks)
    #             results.append(PlateDetection(img=cropped_plate, type=lp_type, coordinate=points))
    #     return results


