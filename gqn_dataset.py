import bisect
from typing import TypeVar, List, Literal
import collections
import os
import numpy as np
import io
import math
from PIL import Image
import torch
import gzip
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset
import random
import itertools
import pytorch_lightning as pl


DATASET_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'gqn')
Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

T_co = TypeVar('T_co', covariant=True)


class MappedDataset(Dataset):
    def __init__(self, inner, transform=None):
        self._transform = transform
        self.inner = inner

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        x = self.inner[idx]
        if self._transform is not None:
            x = self._transform(x)
        return x

    def map(self, transform):
        if self._transform is not None:
            def transform_fn(x):
                return transform(self._transform(x))
        else:
            transform_fn = transform
        return MappedDataset(self.inner, transform_fn)


class EnvironmentDataset(Dataset[T_co]):
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __init__(self, environment_sizes) -> None:
        self.cumulative_sizes = self.cumsum(environment_sizes)

    def __len__(self):
        return self.cumulative_sizes[-1]

    @property
    def num_environments(self):
        return len(self.cumulative_sizes)

    def __getitem__(self, idx):
        def map_index(idx):
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            return (dataset_idx, sample_idx)
        if isinstance(idx, (list, tuple)):
            indices = list(map(map_index, idx))
            sorted_indices = sorted(enumerate(indices), key=lambda x: x[1][0])
            images = [None] * len(indices)
            cameras = [None] * len(indices)
            for key, group in itertools.groupby(sorted_indices, key=lambda x: x[1][0]):
                backward_indices, idxs = tuple(zip(*[(x[0], x[1][1]) for x in group]))
                cimages, ccameras = self.get_sample(key, list(idxs))
                for i, r, c in zip(backward_indices, cimages, ccameras):
                    images[i] = r
                    cameras[i] = c
            return np.stack(images, 0), np.stack(cameras, 0)
        else:
            dataset_idx, sample_idx = map_index(idx)
            return self.get_sample(dataset_idx, sample_idx)

    def get_environment(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        return bisect.bisect_right(self.cumulative_sizes, idx)

    def environment_range(self, environment_idx):
        start = self.cumulative_sizes[environment_idx - 1] if environment_idx != 0 else 0
        end = self.cumulative_sizes[environment_idx]
        return (start, end)

    def get_sample(self, environment_id, sample_id):
        raise NotImplementedError()


def concatenate_shallow(xs):
    if len(xs) == 0:
        return xs
    template = xs[0]
    if isinstance(template, list):
        return list(zip(*xs))
    if isinstance(template, tuple):
        return tuple(zip(*xs))
    if isinstance(template, dict):
        ret = dict()
        for k in template.keys():
            val = []
            for x in xs:
                val.append(x[k])
            ret[k] = val
        return ret
    return xs


class QuerySingleTargetWrapper(Dataset):
    def __init__(self, inner, max_num_views=5, random_views=True):
        assert isinstance(inner, EnvironmentDataset)
        self.inner = inner
        self.epoch = 0
        self.max_num_views = max_num_views
        self.random_views = random_views

    def __len__(self):
        return len(self.inner)

    def set_epoch(self, epoch):
        self.epoch = epoch

    @ staticmethod
    def pair_numbers(a, b):
        return (a + b) * (a+b+1) // 2 + b

    def __getitem__(self, idx):
        return self.getitem(idx)

    def getitem(self, idx, epoch=None):
        if epoch is None:
            epoch = self.epoch
        environment_idx = self.inner.get_environment(idx)
        env_start, env_end = self.inner.environment_range(environment_idx)

        gen = random.Random(self.pair_numbers(epoch, idx))
        num_views = gen.randint(1, self.max_num_views)
        if not self.random_views:
            num_views = self.max_num_views
        context_idx = gen.sample(range(env_start, env_end - 1), num_views)

        # Skip idx
        context_idx = [x + 1 if x >= idx else x for x in context_idx]
        image, pose = self.inner[context_idx + [idx]]
        return (image[:-1], pose[:-1]), (image[-1], pose[-1])


class SizeAdjustedDataset(Dataset):
    def __init__(self, inner, size, strict_size=False):
        self.inner = inner
        self._num_samples_per_item = math.ceil(size / len(self.inner))
        if strict_size:
            self._size = size
        else:
            self._size = self._num_samples_per_item * len(self.inner)
        self.epoch = 0

    def __len__(self):
        return self._size

    def set_epoch(self, epoch):
        self.epoch = epoch

    @ property
    def num_samples_per_item(self):
        return self._num_samples_per_item

    def __getitem__(self, idx):
        if not (0 <= idx < len(self)):
            raise StopIteration()

        sub_epoch = idx // len(self.inner)
        idx = idx % len(self.inner)
        return self.inner.getitem(idx, epoch=self.epoch * self.num_samples_per_item + sub_epoch)


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
        sequence_size=15))


def transform_viewpoint(v):
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)

    return v_hat


def split_name(dataset_name: str):
    split = dataset_name.rindex('-')
    return dataset_name[:split], dataset_name[split + 1:]


class GQNDataset(EnvironmentDataset):
    def __init__(self, name, transform=None, target_transform=None, use_packed=True):
        name, split = split_name(name)
        self.root_dir = os.path.join(DATASET_PATH, f'{name}-th', split)
        self.transform = transform
        self.target_transform = target_transform
        self.use_packed = use_packed
        info = _DATASETS[name]
        super().__init__([info.sequence_size] * len(os.listdir(self.root_dir)))

    def get_sample(self, environment_idx, idx):
        if self.use_packed:
            scene_path = os.path.join(self.root_dir, "{}.pt.gz".format(environment_idx))
            with gzip.open(scene_path, 'rb') as f:
                data = torch.load(f)
        else:
            scene_path = os.path.join(self.root_dir, "{}.pt".format(environment_idx))
            data = torch.load(scene_path)

        def byte_to_tensor(x): return ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))
        images = torch.stack([byte_to_tensor(frame) for frame in data.frames[idx]])
        viewpoints = torch.from_numpy(data.cameras[:, idx])
        # images = torch.stack([byte_to_tensor(frame) for frame in data['frames'][idx]])
        # viewpoints = torch.from_numpy(data['cameras'][idx])
        viewpoints = viewpoints.view(-1, 5)
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)
        return images, viewpoints


def load_gqn_dataset(name):
    dataset_name, split = split_name(name)
    if "jaco" in dataset_name:
        max_num_views = 7
    elif "mazes" in dataset_name:
        max_num_views = 20
    elif "shepard" in dataset_name:
        max_num_views = 14
    elif 'rooms' in dataset_name:
        max_num_views = 5
    else:
        raise ValueError(f'Dataset {name} is not supported')

    def transform_batch(batch):
        context, query = batch
        q_img, q_pose = query
        c_img, c_pose = context
        return dict(query_image=q_img, query_pose=q_pose, context_images=c_img, context_poses=c_pose)

    dataset = GQNDataset(name, target_transform=transform_viewpoint)
    dataset = QuerySingleTargetWrapper(dataset, max_num_views=max_num_views, random_views=False)
    dataset = MappedDataset(dataset, transform_batch)
    return dataset


DatasetName = Literal[tuple(_DATASETS.keys())]


if __name__ == '__main__':
    d = load_gqn_dataset('shepard_metzler_7_parts-train')
    d[0]
    print({k: v.shape for k, v in d[0].items()})
