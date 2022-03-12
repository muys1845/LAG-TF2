import os.path
import tensorflow as tf
import glob
import json
from random import shuffle

dataset_dir = './datasets'
os.makedirs(dataset_dir, exist_ok=True)
assert os.path.exists('./records.json'), 'records file doesn\'t exist'
with open('./records.json', 'r') as f:
    records = json.load(f)
image_formats = ['jpg', 'jpeg', 'png', 'bmp']


class TrainPhase:
    def __init__(self, nimg_start, nimg_stop, lod_start, lod_stop):
        assert 0 <= lod_stop - lod_start <= 1
        assert nimg_start < nimg_stop
        self.nimg_start = nimg_start
        self.nimg_stop = nimg_stop
        self.lod_start = lod_start
        self.lod_stop = lod_stop

    @tf.function
    def lod(self, nimg_cur):  # between lod_start and lod_stop
        if self.lod_start == self.lod_stop:
            return float(self.lod_stop)
        return self.lod_start + (nimg_cur - self.nimg_start) / (self.nimg_stop - self.nimg_start)


class Dataset:
    def __init__(self, data_name, crop=(30, 10), size=(128, 128), test_sample=100):
        def preprocess_image(image):
            if image.shape == 4:
                image = image[..., :-1]
            image = image[crop[0]:-crop[0], crop[1]:-crop[1]]
            image = tf.image.resize(image, size)
            image = image / 127.5 - 1  # normalize to [-1, 1] range
            return image

        assert data_name in records, 'dataset {} not found in: records.json'.format(data_name)
        image_dir = records[data_name]
        all_image_paths = []
        for format_ in image_formats:
            all_image_paths += glob.glob(os.path.join('./datasets', image_dir, '*.' + format_))
        assert all_image_paths, 'no image found, recheck your file: records.json'
        shuffle(all_image_paths)

        tfdata = tf.data.Dataset.from_tensor_slices(all_image_paths).map(tf.io.decode_image).map(preprocess_image)
        self.test_data = tfdata.take(test_sample)
        self.train_data = tfdata.skip(test_sample).repeat()
        self.data_name = data_name
