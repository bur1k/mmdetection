import mmcv
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class BrainwashDataset(CustomDataset):
    CLASSES = ('head',)
    WIDTH = 640
    HEIGHT = 480

    def load_annotations(self, ann_file):
        ann_list = mmcv.list_from_file(ann_file)

        data_infos = []
        for line in ann_list:
            if '";' in line:
                continue

            img_path, current_boxes_as_str = line.strip().split(':')

            bboxes, labels = [], []
            parsed_boxes = current_boxes_as_str.replace('),', '').strip(');').strip(').').split(' (')
            for box in parsed_boxes:
                if len(box) != 0:
                    x1_str, y1_str, x2_str, y2_str = box.replace('.0', '').split(',')
                    x1, y1 = int(x1_str), int(y1_str)
                    x2, y2 = int(x2_str), int(y2_str)

                    x1 = 0 if x1 < 0 else x1
                    y1 = 0 if y1 < 0 else y1
                    x2 = x2 if x2 < self.WIDTH else self.WIDTH - 1
                    y2 = y2 if y2 < self.HEIGHT else self.HEIGHT - 1

                    bboxes.append((x1, y1, x2, y2))
                    labels.append(0)  # single class 'head'

            data_infos.append(
                dict(
                    filename=img_path.strip('"'),
                    width=self.WIDTH,
                    height=self.HEIGHT,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32),
                        labels=np.array(labels).astype(np.int64)
                    )
                )
            )

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
