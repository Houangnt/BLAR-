import os
import glob
import argparse

import cv2

from app.controller.BLPR import get_algorithm


def predict(opt):
    algorithm = get_algorithm()
    list_images = []
    for ext in opt.img_exts:
        list_images += glob.glob(os.path.join(opt.data, '*.{}'.format(ext)))
    with open(opt.save_path, 'w') as fw:
        for image_path in list_images:
            img = cv2.imread(image_path)
            image_name = os.path.basename(image_path)
            #image_name = os.path.split(image_path)[-1]
            result = algorithm.process(org_img=img)
            fw.write(f'{image_name}')
            for index, plate in enumerate(result.plates):
                #cv2.imwrite(f'{index}.jpg', plate)
                cv2.imwrite(os.path.join('./results', f'{image_name}'), plate)
            fw.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/images/', help='source folder')
    parser.add_argument('--img_exts', default=['jpg', 'jpeg', 'png'], nargs='+')
    parser.add_argument('--save_path', default='data/result.txt')
    predict(parser.parse_args())
