import os
import webdataset as wds
from functools import partial
import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl


DATASET_PATH = os.path.join(os.path.expanduser(os.environ.get('DATASETS_PATH', '~/datasets')), 'gqn')


def sample_environment(data, sample_size, environment_size, rng=random, shuffle=False, max_samples_per_env=-1):
    if shuffle:
        data = wds.filters.shuffle(bufsize=environment_size, initial=environment_size, rng=rng)(data)

    def get_env_id(sample):
        return sample['__key__'].split('-')[0]
    current_env = None
    env = []
    samples_per_env = 0
    for sample in data:
        env_id = get_env_id(sample)
        if current_env is None or env_id != current_env:
            env = [sample]
            current_env = env_id
            samples_per_env = 0
        else:
            env.append(sample)
        if len(env) >= sample_size:
            samples_per_env += 1
            if max_samples_per_env < 1 or samples_per_env <= max_samples_per_env:
                yield {k: [x[k] for x in env] for k in env[0].keys()}
            env = []
    if len(env) >= sample_size:
        samples_per_env += 1
        if max_samples_per_env < 1 or samples_per_env <= max_samples_per_env:
            yield {k: [x[k] for x in env] for k in env[0].keys()}


def transform_viewpoint(v):
    w, z = torch.split(v, 3, dim=-1)
    y, p = torch.split(z, 1, dim=-1)

    # position, [yaw, pitch]
    view_vector = [w, torch.cos(y), torch.sin(y), torch.cos(p), torch.sin(p)]
    v_hat = torch.cat(view_vector, dim=-1)
    return v_hat


def transform_batch(data, rng=random, target_transform=None):
    if target_transform is None:
        def target_transform(x): return x
    decoder = wds.Decoder([wds.autodecode.gzfilter, wds.autodecode.ImageHandler('torchrgb'), wds.autodecode.basichandlers])
    for cameras, frames in data:
        output = dict()
        context_size = len(cameras[0]) - 1
        idx_size = rng.randrange(context_size + 1)
        output['context_poses'] = target_transform(torch.stack([torch.stack([torch.from_numpy(decoder.decode1('pth', x)) for x in ctx[:max(1, idx_size)]], 0) for ctx in cameras], 0))[:, :idx_size]
        output['context_images'] = torch.stack([torch.stack([decoder.decode1('jpg', x) for x in ctx[:max(1, idx_size)]], 0) for ctx in frames], 0)[:, :idx_size]
        output['query_pose'] = target_transform(torch.stack([torch.from_numpy(decoder.decode1('pth', x[-1])) for x in cameras], 0))
        output['query_image'] = torch.stack([decoder.decode1('jpg', x[-1]) for x in frames], 0)
        yield output


# transform_batch = wds.filters.Curried(transform_batch_)
# sample_environment = wds.filters.Curried(sample_environment_)
_DATASET_INFO = dict(
    mazes=dict(
        max_num_views=20,
        # train_size=1080,
        # test_size=120,
        train_size=1,
        test_size=1,
        sequence_size=300),
    shepard_metzler_5_parts=dict(
        train_size=900,
        test_size=100,
        max_num_views=14,
        sequence_size=15),
    shepard_metzler_7_parts=dict(
        train_size=900,
        test_size=100,
        max_num_views=14,
        sequence_size=15),
    rooms_free_camera_with_object_rotations=dict(
        max_num_views=5,
        train_size=2034,
        test_size=226,
        sequence_size=10),
    rooms_ring_camera=dict(
        max_num_views=5,
        train_size=2160,
        test_size=240,
        sequence_size=10),
    rooms_free_camera_no_object_rotations=dict(
        max_num_views=5,
        train_size=2160,
        test_size=240,
        sequence_size=10),
    jaco=dict(
        max_num_views=7,
        train_size=3600,
        test_size=400,
        sequence_size=11))


def split_name(dataset_name: str):
    split = dataset_name.rindex('-')
    return dataset_name[:split], dataset_name[split + 1:]


def load_gqn_dataset(name, batch_size, seed=42, shuffle=False, target_transform=transform_viewpoint, max_samples_per_environment=-1):
    dataset_name, split = split_name(name)
    assert dataset_name in _DATASET_INFO, f'Dataset {dataset_name} is not supported'
    assert split in ['test', 'train'], f'Split {split} is not supported'
    dataset_info = _DATASET_INFO[dataset_name]
    size = dataset_info[f'{split}_size']
    url = os.path.join(DATASET_PATH, f'{dataset_name}-wd', f'{dataset_name}-{split}-{{000001..{size:06d}}}-of-{size:06d}.tar')
    sample_size = dataset_info['max_num_views'] + 1
    environment_size = dataset_info['sequence_size']
    dataset = wds.Dataset(url)
    rng = random.Random(seed)
    dataset.rng = rng
    dataset.reseed_hook = dataset.reseed_rng
    if shuffle:
        dataset.shard_shuffle = wds.dataset.Shuffler(rng)
    dataset = dataset.pipe(partial(sample_environment, sample_size=sample_size, environment_size=environment_size, shuffle=shuffle, rng=rng, max_samples_per_env=max_samples_per_environment))
    if shuffle:
        dataset = dataset.pipe(wds.filters.shuffle(10000, rng=rng, initial=1000))
    dataset = dataset.to_tuple('camera.pth', 'image.jpg')
    dataset = dataset.batched(batch_size)
    dataset = dataset.pipe(partial(transform_batch, rng=rng, target_transform=transform_viewpoint))
    return dataset


class InfiniteDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.source = None

    def __getstate__(self):
        result = dict(self.__dict__)
        result["source"] = None
        return result

    def __iter__(self):
        if self.source is None:
            self.source = iter(self.dataset)
        while True:
            try:
                sample = next(self.source)
            except StopIteration:
                self.source = iter(self.dataset)
                sample = next(self.source)
            yield sample


class GQNDataModule(pl.LightningDataModule):
    def __init__(self, dataset: str = 'mazes', batch_size: int = 48, seed: int = 42, num_workers: int = 8, max_samples_per_environment: int = -1):
        super().__init__()
        self.num_workers = num_workers
        self.seed = seed
        self.batch_size = batch_size
        self.dataset = dataset
        self.max_samples_per_environment = max_samples_per_environment

    def setup(self, stage=None):
        def shard_selection(shards):
            assert len(shards) >= self.trainer.world_size
            return shards[self.trainer.global_rank::self.trainer.world_size]

        self.train_dataset = load_gqn_dataset(f'{self.dataset}-train', self.batch_size, seed=self.seed, shuffle=True, max_samples_per_environment=self.max_samples_per_environment)
        self.train_dataset.shard_selection = shard_selection
        self.test_dataset = load_gqn_dataset(f'{self.dataset}-test', self.batch_size, seed=self.seed, shuffle=False, max_samples_per_environment=self.max_samples_per_environment)
        self.test_dataset.shard_selection = shard_selection
        self.train_dataset = InfiniteDataset(self.train_dataset)
        self.test_dataset = InfiniteDataset(self.test_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=None)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=None)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, num_workers=self.num_workers, pin_memory=True, batch_size=None)
