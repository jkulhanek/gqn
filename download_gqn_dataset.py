#!/usr/bin/env python
import argparse
import os
import collections
import torch
import gzip
from google.cloud import storage
import tempfile
import tensorflow as tf
from tqdm import tqdm
import json


DATASET_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'gqn')
DatasetInfo = collections.namedtuple(
    'DatasetInfo',
    ['basepath', 'train_size', 'test_size', 'frame_size', 'sequence_size']
)
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])
Query = collections.namedtuple('Query', ['context', 'query_camera'])
TaskData = collections.namedtuple('TaskData', ['query', 'target'])
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


def download_dataset(name, compress, image_size):
    ext = '.pt'
    if compress:
        ext = '.pt.gz'
    path = os.path.join(DATASET_PATH, f'{name}-th')
    os.makedirs(os.path.join(path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(path, 'test'), exist_ok=True)
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket('gqn-dataset')
    dataset_info = _DATASETS[name]
    with open(os.path.join(path, 'info.json'), 'w+') as f:
        json.dump(dataset_info._asdict(), f)
    existing_data = []
    tot = collections.defaultdict(lambda: 0)
    if os.path.exists(os.path.join(path, 'downloaded.txt')):
        existing_data = [x[:-1].split(' ') for x in open(os.path.join(path, 'downloaded.txt'), 'r')]
    if existing_data:
        tot_train, tot_test, existing_data = tuple(zip(*existing_data))
        tot['train'] = max(map(int, tot_train))
        tot['test'] = max(map(int, tot_test))
    with open(os.path.join(path, 'downloaded.txt'), 'a+') as downloaded_f:
        for blob in tqdm(bucket.list_blobs(prefix=f'{name}/'), total=dataset_info.train_size + dataset_info.test_size):
            if blob.name in existing_data:
                continue
            rest_path = blob.name[len(f'{name}/'):]
            split = rest_path[:rest_path.index('/')]

            def save_file(f):
                # Read file and preprocess
                engine = tf.data.TFRecordDataset(f)
                i = 0
                for i, raw_data in enumerate(engine):
                    file_path = os.path.join(path, split, f'{tot[split]+i}{ext}')
                    # p = Process(target=convert_raw_to_numpy, args=(dataset_info, raw_data, file_path, True))
                    # p.start()
                    # p.join()
                    convert_raw_to_numpy(dataset_info, raw_data, file_path, image_size)
                tot[split] += i
                print(f'{tot["train"]} {tot["test"]} {blob.name}', file=downloaded_f)
                downloaded_f.flush()

            if os.path.exists(os.path.join(DATASET_PATH, 'gqn', blob.name)):
                save_file(os.path.join(DATASET_PATH, 'gqn', blob.name))
            else:
                with tempfile.NamedTemporaryFile('wb+') as f:
                    blob.download_to_file(f)
                    f.flush()
                    f.seek(0)
                    save_file(f.name)
    print(f'Dataset {name} downloaded')


def preprocess_frames(dataset_info, example, image_size=64):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    if image_size != dataset_info.frame_size:
        frames = tf.stack([tf.image.decode_jpeg(x) for x in frames], 0)
        frames = tf.image.convert_image_dtype(frames, dtype=tf.float32)
        frames = tf.image.resize(frames, [image_size, image_size])
        frames = tf.image.convert_image_dtype(frames, dtype=tf.uint8)
        frames = tf.stack([tf.image.encode_jpeg(x) for x in frames], 0)
    return frames.numpy()


def preprocess_cameras(dataset_info, example, raw=True):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [dataset_info.sequence_size, 5])
    if not raw:
        pos = raw_pose_params[:, 0:3]
        yaw = raw_pose_params[:, 3:4]
        pitch = raw_pose_params[:, 4:5]
        cameras = tf.concat(
            [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=-1)
        return cameras.numpy()
    else:
        return raw_pose_params.numpy()


def _get_dataset_files(dataset_info, mode, root):
    """Generates lists of files for a given dataset version."""
    basepath = dataset_info.basepath
    base = os.path.join(root, basepath, mode)
    files = sorted(os.listdir(base))
    return [os.path.join(base, file) for file in files]


def encapsulate(frames, cameras):
    return dict(cameras=cameras, frames=frames)


def convert_raw_to_numpy(dataset_info, raw_data, path, image_size=None):
    feature_map = {
        'frames': tf.io.FixedLenFeature(
            shape=dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.io.FixedLenFeature(
            shape=[dataset_info.sequence_size * 5],
            dtype=tf.float32)
    }
    example = tf.io.parse_single_example(raw_data, feature_map)
    frames = preprocess_frames(dataset_info, example, image_size=image_size)
    cameras = preprocess_cameras(dataset_info, example)
    scene = encapsulate(frames, cameras)
    if path.endswith('.gz'):
        with gzip.open(path, 'wb') as f:
            torch.save(scene, f)
    else:
        torch.save(scene, path)


def show_frame(frames, scene, views):
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.imshow(frames[scene, views])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=list(_DATASETS.keys()) + ['all'])
    parser.add_argument('--decompress', action='store_true', help='Decompress for faster training')
    parser.add_argument('--image-size', default=64, type=float)
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'all':
        for k in _DATASETS.keys():
            download_dataset(k, not args.decompress, args.image_size)
    else:
        download_dataset(dataset, not args.decompress, args.image_size)
