import json
import cv2
import os
from basicsr.utils import img2tensor
from utils import move
import cv2

class PointDataset():
    def __init__(self, base_path: str, base_num: int, save_path=None):
        # base_path: coco_path -> 'COCO'
        base_path = base_path if base_path.endswith('/') else base_path + '/'
        save_path = ('' if save_path == None else base_path) + 'total_data/'

        if os.path.exists(save_path):
            for file in os.listdir(save_path):
                file_path = os.path.join(save_path, file)
                os.remove(file_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        move(base_path, save_path, base_num)

        ori_img = save_path + 'ori_img/'
        point_img = save_path + 'point_img/'

        prompt = save_path + 'prompts/prompts.txt'
        prompts = open(prompt, 'a').readlines()

        ori_list, point_list = os.listdir(ori_img), os.listdir(point_img)
        assert len(ori_list) == len(point_list), 'length unequal error-1'
        assert len(ori_list) == len(prompts), 'length unequal error-2'

        self.ori_imgs = ori_list
        self.point_imgs = point_list
        self.prompts = prompts

    def __getitem__(self, idx):
        ori_img = cv2.imread(self.ori_imgs[idx])
        ori_img = img2tensor(ori_img, bgr2rgb=True, float32=True) / 255.

        point_img = cv2.imread(self.point_imgs[idx])
        point_img = img2tensor(point_img, bgr2rgb=True, float32=True) / 255.

        prompt = self.prompts[idx]

        return {
            'ori_img': ori_img,
            'poi_img': point_img,
            'prompt': prompt
        }

    def __len__(self):
        return len(self.ori_imgs)
