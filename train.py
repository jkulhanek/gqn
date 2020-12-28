#!/bin/env python
import logging
import argparse
import multiprocessing
from utils import setup_logging
import pytorch_lightning as pl
from model import GQNModel
from utils import add_arguments, bind_arguments
from gqn_dataset import DatasetName, load_gqn_dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch
import logging_utils
from web_dataset import GQNDataModule


def build_data(
        batch_size: int = 48,
        dataset: DatasetName = 'shepard_metzler_7_parts',
        num_workers: int = 8,
        seed=42):
    rng = torch.Generator()
    rng.manual_seed(seed)

    def sample_context(batch):
        batch = default_collate(batch)
        context_size = batch['context_images'].shape[1]
        idx_size = torch.randint(context_size + 1, (1,), generator=rng).item()
        batch['context_poses'] = batch['context_poses'][:, :idx_size]
        batch['context_images'] = batch['context_images'][:, :idx_size]
        return batch

    if num_workers > multiprocessing.cpu_count():
        logging.warning(f'Not enough workers available {num_workers} > {multiprocessing.cpu_count()}')
    train_dataset = load_gqn_dataset(f'{dataset}-train')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=sample_context,
        pin_memory=True)

    test_dataset = load_gqn_dataset(f'{dataset}-test')
    test_dataloader = DataLoader(
        test_dataset,
        batch_size,
        num_workers=num_workers,
        collate_fn=sample_context,
        pin_memory=True)
    return (train_dataloader, test_dataloader)


# NOTE: we use different number of total steps, because we run the model on 8GPUs with 10 times the batch size
def build_trainer(
        total_steps: int = 2 * 10 ** 5,
        epochs: int = 50,
        num_gpus: int = 8,
        num_nodes: int = 1,
        profile: bool = False,
        log_graph: bool = False,
        gradient_clip_val: float = 0.0,
        fp16: bool = False,
        wandb: bool = True):
    if wandb:
        logger = logging_utils.WandbLogger(log_model=True, log_graph=log_graph)
    else:
        logger = pl.loggers.TensorBoardLogger(save_dir='gqn', log_graph=log_graph)
    kwargs = dict(num_nodes=num_nodes)
    if num_gpus > 0:
        kwargs.update(dict(gpus=num_gpus, accelerator='ddp'))

    # Split training to #epochs epochs
    limit_train_batches = 1 + total_steps // epochs
    if profile:
        profiler = pl.profiler.AdvancedProfiler()
    else:
        profiler = pl.profiler.PassThroughProfiler()
    if fp16:
        kwargs['precision'] = 16
    trainer = pl.Trainer(
        # max_steps=total_steps,
        max_epochs=epochs,
        val_check_interval=1000,
        gradient_clip_val=gradient_clip_val,
        limit_val_batches=10,
        limit_train_batches=limit_train_batches,
        # track_grad_norm=2,
        logger=logger,
        profiler=profiler,
        callbacks=[logging_utils.LogImageCallback(), pl.callbacks.LearningRateMonitor('step')], **kwargs)
    return trainer


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser, GQNModel)
    parser = add_arguments(parser, build_trainer)
    parser = add_arguments(parser, GQNDataModule)
    args = parser.parse_args()
    model = GQNModel(**bind_arguments(args, GQNModel))
    trainer = build_trainer(**bind_arguments(args, build_trainer))
    if hasattr(trainer.logger.experiment, 'wandb_experiment') and \
            hasattr(trainer.logger.experiment.wandb_experiment, 'config'):
        trainer.logger.experiment.wandb_experiment.config.update(args, allow_val_change=True)
    (train_dataloader, test_dataloader) = build_data(**bind_arguments(args, build_data))

    # Set update epoch hook for datasets
    train_dataset = train_dataloader.dataset
    _old_on_train_epoch_start = trainer.on_train_epoch_start

    def on_train_epoch_start(*args, **kwargs):
        train_dataset.inner.set_epoch(trainer.current_epoch)
        _old_on_train_epoch_start(*args, **kwargs)
    trainer.on_train_epoch_start = on_train_epoch_start

    # Start the training
    datamodule = GQNDataModule(**bind_arguments(args, GQNDataModule))
    trainer.fit(model, datamodule=datamodule)

    # trainer.test(model, test_dataloader)


if __name__ == '__main__':
    main()
