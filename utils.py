import pytorch_lightning as pl


class LogImageCallback(pl.Callback):
    def __init__(self, num_validation_images=32):
        self.num_validation_images = num_validation_images
        super().__init__()

    def on_validation_batch_end(self, trainer, model, model_output, batch, batch_idx, *args, **kwargs):
        if 'image' not in model_output:
            return

        if trainer.global_rank != 0:
            return

        batch_size = trainer.val_dataloaders[0].batch_size
        generated_images = batch_idx * batch_size
        if generated_images >= self.num_validation_images:
            return

        tar_image = batch['query_image']
        gen_image = model_output['generated_image']
        rec_image = model_output['reconstructed_image']
        tar_image = tar_image[:(self.num_validation_images - generated_images)]
        gen_image = gen_image[:(self.num_validation_images - generated_images)]
        rec_image = rec_image[:(self.num_validation_images - generated_images)]
        experiments = trainer.logger.experiment
        if not isinstance(experiments, list):
            experiments = [experiments]
        for experiment in experiments:
            if hasattr(experiment, 'add_image'):
                experiment.add_images('test_generation', gen_image, global_step=trainer.global_step)
                experiment.add_images('test_reconstruction', rec_image, global_step=trainer.global_step)
                experiment.add_images('test_ground_truth', tar_image, global_step=trainer.global_step)
