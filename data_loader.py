from glob import glob
import os
import xml.etree.ElementTree as ET
import random

import cv2


class Dataset(object):
    def __init__(self, xmls_path, images_path, positive_classes):
        self.xml_files = glob(os.path.join(xmls_path, "*.xml"))
        self.image_files = glob(os.path.join(images_path, "*.png"))
        self.image_path_map = {os.path.basename(p).replace(".png", ''): p for p in self.image_files}
        self.positive_classes = positive_classes

    def get_positive_data(self):
        images = []
        labels = []
        for xml_fn in self.xml_files:
            info = self.get_info_from(xml_fn)
            base_fn = os.path.basename(xml_fn).replace(".xml", '')
            img_data = self.get_image_slices(self.image_path_map[base_fn], info)
            for i, img in enumerate(img_data):
                images.append(img)
                labels.append(info[i][0])
        return labels, images

    def get_negative_data(self, size=(300, 300)):
        images = []
        labels = []
        for xml_fn in self.xml_files:
            info = self.get_info_from(xml_fn)
            if not info:
                continue
            positive_zones = [x[2:] for x in info]
            x1, y1, x2, y2 = self.calculate_negative_offsets(info[0][1], size, positive_zones)
            base_fn = base_fn = os.path.basename(xml_fn).replace(".xml", '')

            img_slice = self.get_image_slices(self.image_path_map[base_fn], [['', '', x1, y1, x2, y2]])
            images.extend(img_slice)
            labels.append('negative')
        return labels, images

    def calculate_negative_offsets(self, img_size, slice_size, positive_zones):
        x1 = random.randint(0, img_size[0])
        y1 = random.randint(0, img_size[1])
        x2 = x1 + slice_size[0]
        y2 = y1 + slice_size[1]
        for pzone in positive_zones:
            if x1 > pzone[2] or pzone[0] > x2:
                continue
            elif y2 > pzone[3] or pzone[1] < y1:
                continue
            else:
                self.calculate_negative_offsets(img_size, slice_size, positive_zones)
        return x1, y1, x2, y2

    def get_info_from(self, xml_filepath):
        tree = ET.parse(xml_filepath)
        root = tree.getroot()
        data = []
        size = root.find("size")
        height = int(size.find('width').text)
        width = int(size.find('height').text)
        for node in root.iterfind("object"):
            name = node.find('name').text
            bndbox = node.find("bndbox")
            xmin = bndbox.find("xmin").text
            xmax = bndbox.find("xmax").text
            ymax = bndbox.find("ymax").text
            ymin = bndbox.find("ymin").text
            if name in self.positive_classes:
                data.append((name, (width, height), int(xmin), int(ymin), int(xmax), int(ymax)))
        return data

    def get_image_slices(self, img_filepath, data):
        img = cv2.imread(img_filepath)
        if img is None:
            print("Cannot open {}".format(img_filepath))
            return []
        img_data = []
        for _, _, x1, y1, x2, y2 in data:
            img_data.append(img[int(y1):int(y2), int(x1):int(x2)])
        return img_data

    def load_data(self):
        pos_labels, pos_images = self.get_positive_data()
        neg_labels, neg_images = self.get_negative_data()
        return pos_labels + neg_labels, pos_images + neg_images




