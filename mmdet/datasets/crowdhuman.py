import os
import json
import mmcv
import imagesize
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CrowdHumanDataset(CustomDataset):
    CLASSES = ('head',)

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)

        data_infos = []
        for line in ann_list:
            data = json.loads(line)

            rel_img_path = self.find_proper_dir(data['ID'])
            width, height = imagesize.get(os.path.join(self.img_prefix, rel_img_path))

            bboxes, labels = [], []
            for gt_box in data['gtboxes']:
                if gt_box['tag'] == 'person':
                    if 'ignore' in gt_box['head_attr']:
                        if gt_box['head_attr']['ignore'] == 1:
                            continue

                    x1, y1, w, h = gt_box['hbox']

                    if x1 >= width or y1 >= height:
                        continue

                    x1 = 0 if x1 < 0 else x1
                    y1 = 0 if y1 < 0 else y1
                    w = w if x1 + w < width else width - x1
                    h = h if y1 + h < height else height - y1

                    bboxes.append((x1, y1, x1 + w, y1 + h))
                    labels.append(0)  # single class 'head'

            data_infos.append(
                dict(
                    filename=rel_img_path,
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64)
                    )
                )
            )

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']

    def find_proper_dir(self, name):
        possible_dirs = ['CrowdHuman_train01', 'CrowdHuman_train02', 'CrowdHuman_train03', 'CrowdHuman_val']
        for possible_dir in possible_dirs:
            abs_path = os.path.join(self.img_prefix, possible_dir, name + '.jpg')
            if os.path.exists(abs_path):
                return os.path.join(possible_dir, name + '.jpg')
