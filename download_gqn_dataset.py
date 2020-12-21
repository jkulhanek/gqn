import argparse
import os
import collections
import torch
import gzip
from google.cloud import storage
from multiprocessing import Process
import tempfile
import tensorflow as tf
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


def download_dataset(name):
    path = os.join(DATASET_PATH, f'{name}-th')
    os.makedirs(os.path.join(path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(path, 'test'), exist_ok=True)
    storage_client = storage.Client.create_anonymous_client()
    bucket = storage_client.bucket('gqn-dataset')
    dataset_info = _DATASETS[name]
    with open(os.path.join(path, 'info.json'), 'w+') as f:
        json.dump(dataset_info.__dict__, f)
    existing_data = []
    if os.path.exists(os.path.join(path, 'downloaded.txt')):
        existing_data = [x[:-1] for x in open(os.path.join(path, 'downloaded.txt'), 'r')]
    tot = collections.defaultdict(lambda: 0)
    with open(os.path.join(path, 'downloaded.txt'), 'a+') as downloaded_f:
        for blob in bucket.list_blobs(prefix=f'{name}/'):
            if blob.path in existing_data:
                continue
            rest_path = blob.path[:len(f'{name}/')]
            split = rest_path[:rest_path.index('/')]

            def save_file(f):
                # Read file and preprocess
                engine = tf.python_io.tf_record_iterator(f)
                for i, raw_data in enumerate(engine):
                    file_path = os.path.join(path, split, f'{tot.get(split)+i}.pt.gz')
                    print(f' [-] converting scene {rest_path}-{i} into {path}')
                    p = Process(target=convert_raw_to_numpy, args=(dataset_info, raw_data, file_path, True))
                    p.start()
                    p.join()
                tot[split] += i
                print(blob.path, file=downloaded_f)
                downloaded_f.flush()

            if os.path.exists(os.path.join(DATASET_PATH, 'gqn', blob.path)):
                with open(os.path.join(DATASET_PATH, 'gqn', blob.path), 'rb') as f:
                    save_file(f)
            else:
                with tempfile.TemporaryFile('wb+') as f:
                    blob.download_to_file(f)
                    f.flush()
                    f.seek(0)
                    save_file(f)


def _convert_frame_data(jpeg_data):
    decoded_frames = tf.image.decode_jpeg(jpeg_data)
    return tf.image.convert_image_dtype(decoded_frames, dtype=tf.float32)


def preprocess_frames(dataset_info, example, jpeg='False'):
    """Instantiates the ops used to preprocess the frames data."""
    frames = tf.concat(example['frames'], axis=0)
    if not jpeg:
        frames = tf.map_fn(_convert_frame_data, tf.reshape(frames, [-1]), dtype=tf.float32, back_prop=False)
        dataset_image_dimensions = tuple([dataset_info.frame_size] * 2 + [3])
        frames = tf.reshape(frames, (-1, dataset_info.sequence_size) + dataset_image_dimensions)
        if (64 and 64 != dataset_info.frame_size):
            frames = tf.reshape(frames, (-1,) + dataset_image_dimensions)
            new_frame_dimensions = (64,) * 2 + (3,)
            frames = tf.image.resize_bilinear(frames, new_frame_dimensions[:2], align_corners=True)
            frames = tf.reshape(frames, (-1, dataset_info.sequence_size) + new_frame_dimensions)
    return frames


def preprocess_cameras(dataset_info, example, raw):
    """Instantiates the ops used to preprocess the cameras data."""
    raw_pose_params = example['cameras']
    raw_pose_params = tf.reshape(
        raw_pose_params,
        [-1, dataset_info.sequence_size, 5])
    if not raw:
        pos = raw_pose_params[:, :, 0:3]
        yaw = raw_pose_params[:, :, 3:4]
        pitch = raw_pose_params[:, :, 4:5]
        cameras = tf.concat(
            [pos, tf.sin(yaw), tf.cos(yaw), tf.sin(pitch), tf.cos(pitch)], axis=2)
        return cameras
    else:
        return raw_pose_params


def _get_dataset_files(dataset_info, mode, root):
    """Generates lists of files for a given dataset version."""
    basepath = dataset_info.basepath
    base = os.path.join(root, basepath, mode)
    files = sorted(os.listdir(base))
    return [os.path.join(base, file) for file in files]


def encapsulate(frames, cameras):
    return Scene(cameras=cameras, frames=frames)


def convert_raw_to_numpy(dataset_info, raw_data, path, jpeg=False):
    feature_map = {
        'frames': tf.FixedLenFeature(
            shape=dataset_info.sequence_size, dtype=tf.string),
        'cameras': tf.FixedLenFeature(
            shape=[dataset_info.sequence_size * 5],
            dtype=tf.float32)
    }
    example = tf.parse_single_example(raw_data, feature_map)
    frames = preprocess_frames(dataset_info, example, jpeg)
    cameras = preprocess_cameras(dataset_info, example, jpeg)
    with tf.train.SingularMonitoredSession() as sess:
        frames = sess.run(frames)
        cameras = sess.run(cameras)
    scene = encapsulate(frames, cameras)
    with gzip.open(path, 'wb') as f:
        torch.save(scene, f)


def show_frame(frames, scene, views):
    import matplotlib
    matplotlib.use('qt5agg')
    import matplotlib.pyplot as plt
    plt.imshow(frames[scene, views])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', choices=list(_DATASETS.keys()) + 'all')
    args = parser.parse_args()
    dataset = args.dataset
    if dataset == 'all':
        for k in _DATASETS.keys():
            download_dataset(k)
    else:
        download_dataset(dataset)
