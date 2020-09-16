import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import argparse
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

from detectron2.evaluation.coco_evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine.defaults import build_model
from detectron2.checkpoint import DetectionCheckpointer


class Predictor():
    def __init__(self, config):
        self.config = config
        self.dataset_name_train = "custom_train"
        self.dataset_name_test = "custom_test"

        self.cfg = self.init_model_cfg()

    def init_model_cfg(self):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(self.config.model_zoo))
        cfg.DATASETS.TRAIN = (self.dataset_name_train,)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config.model_zoo)
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
        cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.config.n_class

        cfg.OUTPUT_DIR = self.config.output_dir

        DatasetCatalog._REGISTERED.pop(self.dataset_name_train, None)
        MetadataCatalog._NAME_TO_META.pop(self.dataset_name_train, None)

        train_data_dir = self.config.train_data_dir
        register_coco_instances(self.dataset_name_train, {}, f'{train_data_dir}/annotations.json', f'{train_data_dir}')

        cfg.MODEL.WEIGHTS = self.config.model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.config.score_threshold # set the testing threshold for this model
        cfg.DATASETS.TEST = (self.dataset_name_train, )
        cfg.MODEL.DEVICE = self.config.device_name

        return cfg

    def is_img(self, file):
        _, ext = os.path.splitext(file)
        return ext in ['.png', '.jpg']

    def predict(self):
        predictor = DefaultPredictor(self.cfg)

        test_data_dir = self.config.test_data_dir
        for file in os.listdir(test_data_dir):
            if not self.is_img(file):
                continue
            
            print(file)
            im = cv2.imread(os.path.join(test_data_dir, file))
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=MetadataCatalog.get(self.dataset_name_train), 
                           scale=0.8
            )
            print(outputs["instances"])
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            output_img = v.get_image()[:, :, ::-1] 
            output_path = os.path.join(self.config.output_dir, file)
            cv2.imwrite(output_path, output_img)

    def evaluate(self):
        test_data_dir = self.config.test_data_dir
        register_coco_instances(self.dataset_name, {}, f'{test_data_dir}/annotations.json', f'{test_data_dir}')

        model = build_model(self.cfg)
        DetectionCheckpointer(model).load(self.cfg.MODEL.WEIGHTS) 

        evaluator = COCOEvaluator(self.dataset_name, self.cfg, False, output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, self.dataset_name)
        result = inference_on_dataset(model, val_loader, evaluator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('model_path', help='trained model file path')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    parser.add_argument('--output_dir', default='data/output', help='path to output directory')
    parser.add_argument('--train_data_dir', default='data/train', help='path to training data directory')
    parser.add_argument('--test_data_dir', default='data/test', help='path to test data directory')
    parser.add_argument('--model_zoo', default='COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml')
    parser.add_argument('--score_threshold', type=float, default=0.7)
    parser.add_argument('--n_class', type=int, default=1)
    args = parser.parse_args()

    is_cpu = args.cpu or not torch.cuda.is_available()
    args.device_name = "cpu" if is_cpu else "cuda"
    args.device = torch.device(args.device_name)

    predictor = Predictor(args)
    if args.eval:
        predictor.evaluate()
        sys.exit()

    predictor.predict()

