import os
import cv2
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from collections import namedtuple


class ColorPalette:
    def __init__(self):
        self.colorMap = [[255, 0, 0],
                         [0, 255, 0],
                         [0, 0, 255],
                         [80, 128, 255],
                         [255, 230, 180],
                         [255, 0, 255],
                         [0, 255, 255],
                         [100, 0, 0],
                         [0, 100, 0],
                         [255, 255, 0],
                         [50, 150, 0],
                         [200, 255, 255],
                         [255, 200, 255],
                         [128, 128, 80],
                         [0, 50, 128],
                         [0, 100, 100],
                         [0, 255, 128],
                         [0, 128, 255],
                         [255, 0, 128],
                         [128, 0, 255],
                         [255, 128, 0],
                         [128, 255, 0]]

        self.classMap = ['living room',
                         'kitchen',
                         'bedroom',
                         'bathroom',
                         'closet',
                         'balcony',
                         'corridor',
                         'dining room',
                         'laundry room']

    def classToInt(self, row_label):
        idx_num = len(self.classMap) + 1
        for idx in range(len(self.classMap)):
            if row_label == self.classMap[idx]:
                idx_num = idx + 1
        return idx_num

    def classToStr(self, row_label):
        if not isinstance(row_label, int):
            row_label = int(row_label)
        if row_label > len(self.classMap):
            return 'Other'
        else:
            return self.classMap[row_label-1]

    def getColor(self, classes, is_random=False):
        if not is_random:
            color_idx = self.classToInt(classes)
            return self.colorMap[color_idx]
        else:
            return random.choice(self.colorMap)

    def indexColor(self, class_idx):
        if not isinstance(class_idx, int):
            class_idx = int(class_idx)
        return self.colorMap[class_idx]


class GenerateImage:
    def __init__(self, csv_file, save_dir):
        self.save_dir = save_dir
        self.csv_file = csv_file
        self.color_map = ColorPalette()

    def getBaseImg(self, img_size):
        r = np.full((*img_size, 1), 255, np.uint8)
        g = np.full((*img_size, 1), 255, np.uint8)
        b = np.full((*img_size, 1), 255, np.uint8)
        return np.concatenate((r, g, b), axis=2)

    def splitGroup(self, df, group):
        data = namedtuple('data', group + ['object'])
        gb = df.groupby(group)
        return [data(filename, height, width, gb.get_group(x))
                for (filename, height, width), x in zip(gb.groups.keys(), gb.groups)]

    def concatRooms(self, group, text=False):
        filename = os.path.join(self.save_dir, str(group.filename) + '.png')
        img = self.getBaseImg([group.height, group.width])
        for _, room in group.object.iterrows():
            cv2.rectangle(img, (room['x'], room['y']), (room['x']+room['w'], room['y']+room['h']),
                          self.color_map.getColor(room['class']), -1)
            if text:
                cv2.putText(img, room['class'], (room['x'], room['y'] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_map.getColor(room['class']), 2)

        cv2.imwrite(filename, img)

    def generateImg(self):
        rooms_info = pd.read_csv(self.csv_file, header=0)
        grouped = self.splitGroup(rooms_info, ['filename', 'height', 'width'])
        for group in tqdm(grouped, desc='Create Room Information in PNG image'):
            self.concatRooms(group, text=True)


class GenerateTfrecord:
    def __init__(self, csv_file, save_dir):
        self.csv_file = csv_file
        self.save_dir = save_dir
        self.color_map = ColorPalette()

    def splitGroup(self, df, group):
        data = namedtuple('data', group + ['object'])
        gb = df.groupby(group)
        return [data(filename, height, width, gb.get_group(x))
                for (filename, height, width), x in zip(gb.groups.keys(), gb.groups)]

    def bytesFeature(self, values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def floatFeature(self, values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def int64Feature(self, values):
        if not isinstance(values, (tuple, list)):
            values = [values]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


    def createTfValue(self, group):
        x, y, w, h, classes, classes_text = [], [], [], [], [], []

        for _, row in group.object.iterrows():
            x.append(int(row['x']) / int(group.width))
            y.append(int(row['y']) / int(group.height))
            w.append(int(row['w']) / int(group.width))
            h.append(int(row['h']) / int(group.height))
            classes_text.append(row['class'].encode('utf8'))
            classes.append(self.color_map.classToInt(row['class']))

        label_dict = {
            'filename': self.bytesFeature(group.filename.encode('utf8')),
            'width': self.floatFeature(group.width),
            'height': self.floatFeature(group.height),
            'object/bbox/x': self.floatFeature(x),
            'object/bbox/y': self.floatFeature(y),
            'object/bbox/w': self.floatFeature(w),
            'object/bbox/h': self.floatFeature(h),
            'object/class/text': self.bytesFeature(classes_text),
            'object/class/label': self.int64Feature(classes),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=label_dict))
        return tf_example

    def createTfrecord(self, mode):
        writer = tf.io.TFRecordWriter(os.path.join(self.save_dir + "{}.record".format(mode)))
        rooms_info = pd.read_csv(self.csv_file, header=0)
        grouped = self.splitGroup(rooms_info, ['filename', 'height', 'width'])
        for group in tqdm(grouped, desc='Create Room Information in Tfrecord'):
            tf_example = self.createTfValue(group)
            writer.write(tf_example.SerializeToString())
        writer.close()


