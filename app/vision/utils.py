import cv2
import numpy as np


def crop_affine(img, landmarks, lp_type):
    lp_transform_dest = [None, np.float32([[0, 0], [125, 0], [125, 32], [0, 32]]),
                         np.float32([[0, 0], [190, 0], [190, 140], [0, 140]])]
    lp_crop_size = [None, (125, 32), (190, 140)]
    h, w, c = img.shape
    points = []
    for i in range(4):
        points.append([landmarks[2 * i] * w, landmarks[2 * i + 1] * h])
    points = np.array(points, np.float32)
    M = cv2.getPerspectiveTransform(points, lp_transform_dest[lp_type])
    dst = cv2.warpPerspective(img, M, lp_crop_size[lp_type])
    return dst


def draw_landmark(img, landmarks):
    width = img.shape[1]
    height = img.shape[0]
    tl = 1 or round(0.002 * (height + width) / 2) + 1  # line/font thickness
    # for land in lands:
    x_arr = []
    y_arr = []
    land = landmarks
    for i in range(len(land)):
        if i % 2 == 0:
            x_arr.append(int(land[i] * width))
        else:
            y_arr.append(int(land[i] * height))

    cv2.line(img, (x_arr[0], y_arr[0]), (x_arr[1], y_arr[1]), (0, 0, 255), 5, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(img, (x_arr[1], y_arr[1]), (x_arr[2], y_arr[2]), (0, 0, 255), 5, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(img, (x_arr[2], y_arr[2]), (x_arr[3], y_arr[3]), (0, 0, 255), 5, cv2.FONT_HERSHEY_SIMPLEX)
    cv2.line(img, (x_arr[3], y_arr[3]), (x_arr[0], y_arr[0]), (0, 0, 255), 5, cv2.FONT_HERSHEY_SIMPLEX)

    # new_width = int(width / 4)
    # new_height = int(height / 4)
    # img = cv2.resize(img, (new_width, new_height))

    return img


def draw_label(image, detection_results):
    coordinates = []
    for detection_result in detection_results:
        for row in detection_result.coordinates:
            coordinates.append(row)
    for coordinate in coordinates:
        dst = cv2.rectangle(image, (coordinate[0], coordinate[1]), (coordinate[2], coordinate[3]), (255, 0, 0), 2)
    return dst


def preprocess_crop_rec(detection_results):
    cropped_images = []
    for detection_result in detection_results:
        for row in detection_result.coordinates:
            cropped_images.append(detection_result.image[row[1]:row[3], row[0]:row[2]])
    return cropped_images


def preprocess_crop_affine(detection_results):
    cropped_images = []
    for detection_result in detection_results:
        for row in detection_result.coordinates:
            cropped_images.append(crop_affine(detection_result.image, row, 2))
    return cropped_images


def preprocess_draw_rec(detection_results):
    bbox = []
    for detection_result in detection_results:
        for row in detection_result.coordinates:
            bbox.append(draw_landmark(detection_result.image, row))
        # bbox.append(draw_label(detection_result.image, detection_results))
    return bbox


def preprocess_draw_yolo(detection_results):
    bbox = []
    for detection_result in detection_results:
        bbox.append(draw_label(detection_result.image, detection_results))
    # bbox.append(draw_label(detection_result.image, detection_results))
    return bbox
