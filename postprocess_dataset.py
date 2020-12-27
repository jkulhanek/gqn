#!/usr/bin/env python
import argparse
import os
import collections
import torch
import gzip
import tensorflow as tf
from tqdm import tqdm


DATASET_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'gqn')
DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
_DATASETS = dict(
    jaco=DatasetInfo(
        basepath='jaco',
        train_size=3600,
        test_size=400,
        frame_size=64,
        sequence_size=11),

    mazes=DatasetInfo(
        basepath='mazes',
        train_size=1080,
        test_size=120,
        frame_size=84,
        sequence_size=300),

    rooms_free_camera_with_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_with_object_rotations',
        train_size=2034,
        test_size=226,
        frame_size=128,
        sequence_size=10),

    rooms_ring_camera=DatasetInfo(
        basepath='rooms_ring_camera',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    rooms_free_camera_no_object_rotations=DatasetInfo(
        basepath='rooms_free_camera_no_object_rotations',
        train_size=2160,
        test_size=240,
        frame_size=64,
        sequence_size=10),

    shepard_metzler_5_parts=DatasetInfo(
        basepath='shepard_metzler_5_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15),

    shepard_metzler_7_parts=DatasetInfo(
        basepath='shepard_metzler_7_parts',
        train_size=900,
        test_size=100,
        frame_size=64,
        sequence_size=15)
)


def find_dataset_size(path):
    usize = 1
    while os.path.exists(os.path.join(path, f'{usize}.pt.gz')):
        usize *= 2
    lsize = usize // 2
    while usize - lsize > 1:
        if os.path.exists(os.path.join(path, f'{(usize + lsize) // 2}.pt.gz')):
            lsize = (usize + lsize) // 2
        else:
            usize = (usize + lsize) // 2
    return usize


def main(name):
    print(f'Starting postprocessing dataset {name}')
    path = os.path.join(DATASET_PATH, f'{name}-th')
    train_total = find_dataset_size(os.path.join(path, 'train'))
    test_total = find_dataset_size(os.path.join(path, 'test'))
    with tqdm(total=train_total + test_total) as progress:
        for i in range(train_total):
            file_path = os.path.join(path, 'train', f'{i}.pt.gz')
            out_file_path = os.path.join(path, 'train', f'{i}.pt')
            with gzip.open(file_path, 'rb') as f:
                data = torch.load(f)
                data['frames'] = preprocess_frames(data['frames'])
                torch.save(data, out_file_path)
            progress.update()

        for i in range(test_total):
            file_path = os.path.join(path, 'test', f'{i}.pt.gz')
            out_file_path = os.path.join(path, 'test', f'{i}.pt')
            with gzip.open(file_path, 'rb') as f:
                data = torch.load(f)
                data['frames'] = preprocess_frames(data['frames'])
                torch.save(data, out_file_path)
            progress.update()


def preprocess_frames(frames):
    image_size = 64
    frames = tf.stack([tf.image.decode_jpeg(x) for x in frames], 0)
    frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)
    frames = tf.image.resize(frames, [image_size, image_size])
    frames = tf.image.convert_image_dtype(frames, dtype=tf.uint8)
    frames = tf.stack([tf.image.encode_jpeg(x) for x in frames], 0)
    return frames.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=list(_DATASETS.keys()) + ['all'])
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'all':
        for k in _DATASETS.keys():
            main(k)
    else:
        main(dataset)
