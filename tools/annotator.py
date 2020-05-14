import cv2
import sys
import os.path
import argparse
import hashlib
from tqdm import tqdm
from pascal_voc_writer import Writer
import xml.etree.ElementTree as ET

import os
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

script_dir = os.path.dirname(os.path.realpath(__file__))

class Annotator():

    CASCADE_FILE = os.path.join(script_dir, 'lbpcascade_animeface.xml')
    KEY_LABEL_MAP = {
        '0': 'asuna',
        '1': 'administrator',
        '2': 'kirito',
        '3': 'alice',
        '4': 'yui',
    }

    def __init__(self, config):
        self.config = config

        target = config.target
        if os.path.isfile(target):
            self.target = target
            self.target_dir = os.path.dirname(target)
            self.output_dir = os.path.splitext(target)[0]
        else:
            self.target = None
            self.target_dir = target
            self.output_dir = os.path.join(self.target_dir, 'annotations')

        if config.output_dir is not None:
            self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        if config.model_detectron:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = config.detectron_num_classes
            cfg.MODEL.DEVICE = 'cpu' if config.cpu else 'cuda'
            cfg.MODEL.WEIGHTS = config.model_detectron
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = config.detectron_score_thresh_test

            self.predictor = DefaultPredictor(cfg)

    def annotate(self, path):
        if not self.config.model_detectron and not os.path.isfile(self.CASCADE_FILE):
            raise RuntimeError("%s: not found" % self.CASCADE_FILE)

        if self.__is_image(path):
            self.annotate_image(path)
            return

    def detect_faces(self, image):
        if self.config.model_detectron:
            outputs = self.predictor(image)
            boxes = outputs['instances'].get('pred_boxes')
            ret = []
            for box in boxes:
                ret.append([
                    int(box[0].item()),
                    int(box[1].item()),
                    int((box[2]-box[0]).item()),
                    int((box[3]-box[1]).item())
                ])
            return ret

        cascade = cv2.CascadeClassifier(self.CASCADE_FILE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        return cascade.detectMultiScale(gray,
                                        # detector options
                                        scaleFactor = 1.1,
                                        minNeighbors = 1,
                                        minSize = (24, 24))

    def annotate_image(self, path, image=None):
        if image is None:
            image = cv2.imread(path, cv2.IMREAD_COLOR)

        faces = self.detect_faces(image)
        print(len(faces))

        output_path = self.__get_output_path(image)
        shape = image.shape
        writer = Writer(output_path, shape[1], shape[0], shape[2])

        saved_count = 0
        for (x, y, w, h) in faces:
            xmax = x + w
            ymax = y + h
            if not self.config.all_faces:
                image_tmp = image.copy()
                cv2.rectangle(image_tmp, (x, y), (xmax, ymax), (0, 0, 255), 2)
                cv2.imshow("annotation", image_tmp)

            label = self.get_label()
            if label is not None:
                writer.addObject(label, x, y, xmax, ymax)
                saved_count = saved_count + 1

        if 0 < saved_count:
            cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            name = os.path.splitext(output_path)
            writer.save('{}.xml'.format(name[0]))

    def get_label(self):
        if self.config.all_faces:
            return 'face'

        key = chr(cv2.waitKey(0))
        if key not in self.KEY_LABEL_MAP:
            return None

        return self.KEY_LABEL_MAP[key]

    def __is_image(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.jpg', '.jpeg', '.png']

    def __is_movie(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.mp4', '.mov']

    def __is_xml(self, path):
        _, ext = os.path.splitext(path)
        return ext in ['.xml']

    def __get_output_path(self, image):
        md5 = hashlib.md5(image).hexdigest()
        return '{}/{}.jpg'.format(self.output_dir, md5, 'jpg')

    def extract_images(self):
        if not self.__is_movie(self.target):
            print('{} is not movie file'.format(self.target))
            return

        cap = cv2.VideoCapture(self.target)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        detector = cv2.AKAZE_create()

        difference = 1000
        _, image = cap.read()
        des_prev = self.__detectAndCompute(image, detector)
        for _ in tqdm(range(frame_count)):
            if not cap.isOpened:
                break

            frame = int(round(cap.get(1)))
            _, image = cap.read()

            if frame % self.config.sampling_frames != 0:
                continue

            des = self.__detectAndCompute(image, detector)
            if des is None:
                continue

            matches = bf.match(des_prev, des)
            dist = [m.distance for m in matches]
            difference = sum(dist) / len(dist)

            if difference > self.config.difference_threshold:
                des_prev = des

                faces = self.detect_faces(image)
                if len(faces) != 0:
                    output_path = self.__get_output_path(image)
                    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        cap.release()

    def __detectAndCompute(self, image, detector):
        image = cv2.resize(image, (640, 360))
        _, des = detector.detectAndCompute(image, None)
        return des

    def extract_labels(self):
        for root, dirs, files in os.walk(self.target_dir):
            for file in files:
                if not self.__is_xml(file):
                    continue

                xml_path = os.path.join(root, file)
                with open(xml_path, 'r') as f:
                    xml = ET.fromstring(f.read())

                for obj in xml.findall('object'):
                    print('{} {}'.format(file, obj.find('name').text))

    def fix(self):
        for root, dirs, files in os.walk(self.target_dir):
            for file in files:
                if not self.__is_xml(file):
                    continue

                xml_path = os.path.join(root, file)
                with open(xml_path, 'r') as f:
                    xml = ET.fromstring(f.read())

                path = xml.find('path').text
                width = int(xml.find('size/width').text)
                height = int(xml.find('size/height').text)

                writer = Writer(path, height, width, 3)
                for obj in xml.findall('object'):
                    xmin = int(obj.find('bndbox/xmin').text)
                    ymin = int(obj.find('bndbox/ymin').text)
                    xmax = int(obj.find('bndbox/xmax').text)
                    ymax = int(obj.find('bndbox/ymax').text)
                    writer.addObject(obj.find('name').text, xmin, ymin, xmax, ymax)
                writer.save(xml_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='annotate images')
    parser.add_argument('target', default='./images', help='path to target file or directory that has target images')
    parser.add_argument('--mv2img', action='store_true', help='extract images from movie file')
    parser.add_argument('--sampling_frames', type=int, default=30, help='sampling image by this frames')
    parser.add_argument('--difference_threshold', type=int, default=20, help='sampling when image difference is over this value from last sampling image')
    parser.add_argument('--extract_labels', action='store_true', help='extract label from annotation file')
    parser.add_argument('--fix', action='store_true', help='')
    parser.add_argument('--all_faces', action='store_true', help='annotate all faces as "face"')
    parser.add_argument('--recursive', action='store_true', help='search target files recursively')
    parser.add_argument('--output_dir', default=None, help='path to output directory')
    parser.add_argument('--model_detectron', default=None, help='path to model file for detectron')
    parser.add_argument('--detectron_score_thresh_test', type=float, default=0.5, help='detect boxes with score more than this')
    parser.add_argument('--detectron_num_classes', type=int, default=1, help='class num in model_detectron')
    parser.add_argument('--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()

    annotator = Annotator(args)

    if args.fix:
        annotator.fix()
        sys.exit()

    if args.mv2img:
        annotator.extract_images()
        sys.exit()

    if args.extract_labels:
        annotator.extract_labels()
        sys.exit()

    if not args.recursive:
        for file in tqdm(os.listdir(args.target)):
            annotator.annotate(os.path.join(args.target, file))
        sys.exit()

    for root, _, files in os.walk(args.target):
        for file in tqdm(files):
            annotator.annotate(os.path.join(root, file))
