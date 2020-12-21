import bisect
from typing import TypeVar
import collections
import os
import io
import math
from PIL import Image
import torch
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import Dataset
import random
import itertools


Context = collections.namedtuple('Context', ['frames', 'cameras'])
Scene = collections.namedtuple('Scene', ['frames', 'cameras'])

T_co = TypeVar('T_co', covariant=True)


class EnvironmentDataset(Dataset[T_co]):
    datasets: List[Dataset[T_co]]
    cumulative_sizes: List[int]

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += l
        return r

    def __init__(self, environment_sizes) -> None:
        self.cumulative_sizes = self.cumsum(environment_sizes)

    def __len__(self):
        return self.cumulative_sizes[-1]

    @property
    def num_environments(self):
        return len(self.cummulative_sizes)

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
            values = [None] * len(indices)
            for group in itertools.groupby(sorted_indices, key=lambda x: x[1][0]):
                backward_indices, idxs = tuple(zip(*[(x[0], x[1][1]) for x in group]))
                results = self.get_sample(group.key, idxs)
                for i, r in zip(backward_indices, results):
                    values[i] = r
            return values
        else:
            dataset_idx, sample_idx = map_index(idx)
            return get_sample(dataset_idx, sample_idx)

    def get_environment(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        return bisect.bisect_right(self.cumulative_sizes, idx)

    def environment_range(self, environment_idx):
        start = self.cummulative_sizes[environment_idx - 1] if environment_idx != 0 else 0
        end = self.cummulative_sizes[environment_idx]
        return (start, end)

    def get_sample(self, environment_id, sample_id):
        raise NotImplementedError()

    @ property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes


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
            res[k] = val
        return ret
    return xs


class QuerySingleTargetWrapper(Dataset):
    def __init__(self, inner, max_num_views=5):
        assert isinstance(inner, EnvironmentDataset)
        self.inner = inner
        self.epoch = 0

    def __len__(self):
        return len(inner)

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
        num_views = random.randint(1, self.max_num_views)
        context_idx = random.sample(range(env_start, env_end - 1), num_views)

        # Skip idx
        context_idx = [x + 1 if x >= idx else x for x in context_idx]
        results = self.inner[context_idx + [idx]]
        return concatenate_shallow(results[:-1]), results[-1]


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


class GQNDataset(EnvironmentDataset):
    def __init__(self, root_dir, name, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        info = _DATASETS[name]
        super().__init__([info['sequence_size']] * len(os.listdir(self.root_dir)))

    def get_sample(self, environment_idx, idx):
        scene_path = os.path.join(self.root_dir, "{}.pt".format(environment_idx))
        data = torch.load(scene_path)
        def byte_to_tensor(x): return ToTensor()(Resize(64)((Image.open(io.BytesIO(x)))))
        images = torch.stack([byte_to_tensor(frame) for frame in data.frames[idx]])
        viewpoints = torch.from_numpy(data.cameras[idx])
        viewpoints = viewpoints.view(-1, 5)
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            viewpoints = self.target_transform(viewpoints)
        return images, viewpoints


def load_gqn_dataset(name, seed=42):
    if name == "Room":
        max_num_views = 5
    elif name == "Jaco":
        max_num_views = 7
    elif D == "Labyrinth":
        max_num_views = 20
    elif D == "Shepard-Metzler":
        max_num_views = 15

    dataset = GQNDataset(root_dir, name, transform, target_transform)
    dataset = QuerySingleTargetWrapper(dataset, max_num_views=max_num_views)
