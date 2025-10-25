import logging
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.dataset.utils import get_dataloaders
from src.trainer.classification import ClassificationTrainer
from src.utils.init import set_random_seed
from src.utils.torch import get_lr_scheduler, get_num_params

logger = logging.getLogger(Path(__file__).name)


@hydra.main(version_base='1.3', config_path='src/configs', config_name='rad-dino')
def main(cfg: DictConfig):
    set_random_seed(cfg.trainer.seed)
    logger.info(f'Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')
    if cfg.trainer.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.trainer.device
    logger.info(f'Using device: {device}')

    dataloaders, batch_transforms = get_dataloaders(cfg, device)

    model = instantiate(cfg.model).to(device)
    logger.info(f'Model:\n{model}')
    num_params = get_num_params(model, trainable_only=True)
    logger.info(f'Trainable parameters: {num_params}')
    model = torch.compile(model)
    optimizer = instantiate(
        cfg.optimizer,
        params=[p for p in model.parameters() if p.requires_grad],
    )
    logger.info(f'Optimizer:\n{optimizer}')
    lr_scheduler = get_lr_scheduler(cfg, optimizer, dataloaders['train'])
    logger.info(f'LR Scheduler: {lr_scheduler}')

    criterion = instantiate(cfg.criterion).to(device)
    logger.info(f'Criterion: {criterion}')
    metrics = instantiate(cfg.metrics)
    logger.info(f'Metrics: {metrics}')

    project_config = OmegaConf.to_container(cfg, resolve=True)
    project_config['model']['num_params'] = num_params
    writer = instantiate(cfg.writer, project_config)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = cfg.trainer.get('epoch_len')
    trainer = ClassificationTrainer(
        cfg=cfg,
        device=device,
        dataloaders=dataloaders,
        model=model,
        criterion=criterion,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        writer=writer,
        skip_oom=cfg.trainer.get('skip_oom', True),
        batch_transforms=batch_transforms,
        epoch_len=epoch_len,
    )

    trainer.train()


if __name__ == '__main__':
    main()
