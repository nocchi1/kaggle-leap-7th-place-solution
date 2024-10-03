from omegaconf import DictConfig
from torch import nn

from src.model.models.conv1d_seq2seq import Conv1DSeq2Seq
from src.model.models.lstm_seq2seq import LSTMSeq2Seq
from src.model.models.transformer_seq2seq import TransformerSeq2Seq
from src.train.optimizer import get_optimizer
from src.train.scheduler import get_scheduler


class ComponentFactory:
    def __init__(self, config: DictConfig):
        self.config = config

    def get_model(self):
        if self.config.task_type == "main":
            if self.config.model_type == "conv1d":
                model = Conv1DSeq2Seq(
                    in_dim=self.config.in_dim,
                    out_dim=self.config.out_dim,
                    hidden_dim=self.config.hidden_dim,
                    block_num=self.config.block_num,
                    kernel_size=self.config.kernel_size,
                    multitask=self.config.multi_task,
                )
            elif self.config.model_type == "lstm":
                model = LSTMSeq2Seq(
                    in_dim=self.config.in_dim,
                    out_dim=self.config.out_dim,
                    hidden_dim=self.config.hidden_dim,
                    block_num=self.config.block_num,
                    scaler_num=self.config.scaler_num,
                    multitask=self.config.multi_task,
                )
            elif self.config.model_type == "transformer":
                model = TransformerSeq2Seq(
                    in_dim=self.config.in_dim,
                    out_dim=self.config.out_dim,
                    hidden_dim=self.config.hidden_dim,
                    trans_num_layers=self.config.trans_num_layers,
                    lstm_block_num=self.config.lstm_block_num,
                    scaler_num=self.config.scaler_num,
                    multitask=self.config.multi_task,
                )
        elif self.config.task_type == "grid_pred":
            pass
        return model

    def get_loss(self):
        if self.config.task_type == "main":
            if self.config.loss_type == "mse":
                loss_fn = nn.MSELoss()
            elif self.config.loss_type == "mae":
                loss_fn = nn.L1Loss()
            elif self.config.loss_type == "huber":
                loss_fn = nn.HuberLoss()
            elif self.config.loss_type == "inverse_huber":
                # loss_fn = InverseHuberLoss()  # [TODO]これを再考したい
                pass
        elif self.config.task_type == "grid_pred":
            loss_fn = nn.CrossEntropyLoss()
        return loss_fn

    def get_optimizer(self):
        optimizer = get_optimizer(
            self.model,
            optimizer_type=self.config.optimizer_type,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            betas=self.config.betas,
        )
        return optimizer

    def get_scheduler(self, steps_per_epoch):
        total_steps = self.config.epochs * steps_per_epoch
        if self.config.scheduler_type == "linear":
            scheduler_args = {
                "num_warmup_steps": self.config.num_warmup_steps,
                "num_training_steps": total_steps,
            }
        elif self.config.scheduler_type == "cosine":
            scheduler_args = {
                "num_warmup_steps": self.config.num_warmup_steps,
                "num_training_steps": total_steps,
                "num_cycles": self.config.num_cycles,
            }
        elif self.config.scheduler_type == "cosine_custom":
            first_cycle_steps = self.config.first_cycle_epochs * steps_per_epoch
            scheduler_args = {
                "first_cycle_steps": first_cycle_steps,
                "cycle_factor": self.config.cycle_factor,
                "num_warmup_steps": self.config.num_warmup_steps,
                "min_lr": self.config.min_lr,
                "gamma": self.config.gamma,
            }
        elif self.config.scheduler_type == "reduce_on_plateau":
            scheduler_args = {
                "mode": self.config.mode,
                "factor": self.config.factor,
                "patience": self.config.patience,
                "min_lr": self.config.min_lr,
            }
        else:
            raise ValueError(f"Invalid scheduler_type: {self.config.scheduler_type}")

        scheduler = get_scheduler(
            self.optimizer, scheduler_type=self.config.scheduler_type, scheduler_args=scheduler_args
        )
        return scheduler
