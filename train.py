#!/bin/env python
import logging
import argparse
import multiprocessing
from utils import setup_logging
from utils import TensorBoardWandbLogger, LogImageCallback
import pytorch_lightning as pl
from model import GQNModel
from utils import add_arguments, bind_arguments
from gqn_dataset import DatasetName, load_gqn_dataset, Scene
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch


def build_data(
        batch_size: int = 36,
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


def build_trainer(
        total_steps: int = 2 * 10 ** 6,
        wandb: bool = True):
    if wandb:
        logger = pl.loggers.WandbLogger()
        logger = [logger, TensorBoardWandbLogger(logger)]
    else:
        logger = [pl.loggers.TensorBoardLogger('logs')]
    trainer = pl.Trainer(
        max_steps=total_steps,
        val_check_interval=10000,
        limit_val_batches=100,
        logger=logger,
        callbacks=[LogImageCallback(), pl.callbacks.LearningRateMonitor('step')])
    return trainer


def main():
    setup_logging()
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser, GQNModel)
    parser = add_arguments(parser, build_trainer)
    parser = add_arguments(parser, build_data)
    args = parser.parse_args()
    model = GQNModel(**bind_arguments(args, GQNModel))
    trainer = build_trainer(**bind_arguments(args, build_trainer))
    (train_dataloader, test_dataloader) = build_data(**bind_arguments(args, build_data))

    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=test_dataloader)

    # trainer.test(model, test_dataloader)


if __name__ == '__main__':
    main()
