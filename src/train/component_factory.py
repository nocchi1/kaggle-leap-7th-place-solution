from omegaconf import DictConfig
from torch import nn
from torch.optim.optimizer import Optimizer

from src.model.models.conv1d import GridPredConv1D, LEAPConv1D
from src.model.models.lstm import LEAPLSTM
from src.model.models.squeezeformer import LEAPSqueezeformer
from src.model.models.transformer import LEAPTransformer
from src.train.loss import LEAPLoss
from src.train.optimizer import get_optimizer
from src.train.scheduler import get_scheduler


class ComponentFactory:
    @staticmethod
    def get_model(config: DictConfig):
        if config.task_type == "main":
            if config.model_type == "conv1d":
                model = LEAPConv1D(
                    in_dim=config.in_dim,
                    out_dim=config.out_dim,
                    hidden_dim=config.hidden_dim,
                    block_num=config.block_num,
                    kernel_size=config.kernel_size,
                    multi_task=config.multi_task,
                )
            elif config.model_type == "lstm":
                model = LEAPLSTM(
                    in_dim=config.in_dim,
                    out_dim=config.out_dim,
                    hidden_dim=config.hidden_dim,
                    block_num=config.block_num,
                    scaler_num=config.scaler_num,
                    multi_task=config.multi_task,
                )
            elif config.model_type == "transformer":
                model = LEAPTransformer(
                    in_dim=config.in_dim,
                    out_dim=config.out_dim,
                    hidden_dim=config.hidden_dim,
                    trans_num_layers=config.trans_num_layers,
                    lstm_block_num=config.lstm_block_num,
                    scaler_num=config.scaler_num,
                    multi_task=config.multi_task,
                )
            elif config.model_type == "squeezeformer":
                model = LEAPSqueezeformer(
                    in_dim=config.in_dim,
                    out_dim=config.out_dim,
                    hidden_dim=config.hidden_dim,
                    block_num=config.block_num,
                    kernel_size=config.kernel_size,
                    multi_task=config.multi_task,
                )
        elif config.task_type == "grid_pred":
            model = GridPredConv1D(
                in_dim=config.in_dim,
                out_dim=config.out_dim,
                hidden_dim=config.hidden_dim,
                block_num=config.block_num,
                kernel_size=config.kernel_size,
            )
        return model

    @staticmethod
    def get_loss(config: DictConfig):
        if config.task_type == "main":
            loss_fn = LEAPLoss(config)
        elif config.task_type == "grid_pred":
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    @staticmethod
    def get_optimizer(config: DictConfig, model):
        optimizer = get_optimizer(
            model,
            optimizer_type=config.optimizer_type,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
        return optimizer

    @staticmethod
    def get_scheduler(config: DictConfig, optimizer: Optimizer, steps_per_epoch: int):
        total_steps = config.epochs * steps_per_epoch
        if config.scheduler_type == "linear":
            scheduler_args = {
                "num_warmup_steps": config.num_warmup_steps,
                "num_training_steps": total_steps,
            }
        elif config.scheduler_type == "cosine":
            scheduler_args = {
                "num_warmup_steps": config.num_warmup_steps,
                "num_training_steps": total_steps,
                "num_cycles": config.num_cycles,
            }
        elif config.scheduler_type == "cosine_custom":
            first_cycle_steps = config.first_cycle_epochs * steps_per_epoch
            scheduler_args = {
                "first_cycle_steps": first_cycle_steps,
                "cycle_factor": config.cycle_factor,
                "num_warmup_steps": config.num_warmup_steps,
                "min_lr": config.min_lr,
                "gamma": config.gamma,
            }
        elif config.scheduler_type == "reduce_on_plateau":
            scheduler_args = {
                "mode": config.mode,
                "factor": config.factor,
                "patience": config.patience,
                "min_lr": config.min_lr,
            }
        else:
            raise ValueError(f"Invalid scheduler_type: {config.scheduler_type}")

        scheduler = get_scheduler(optimizer, scheduler_type=config.scheduler_type, scheduler_args=scheduler_args)
        return scheduler
