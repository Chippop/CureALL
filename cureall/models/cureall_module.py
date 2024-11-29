from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torch.nn import functional as F

from cureall.utils.losses import binary_top_k_loss
from cureall.utils.metrics import CureALLMetrics
from cureall.utils.model_utils import convert_unimol_ckpt

from .modeling_uce import UCEConfig, UCEModel
from .modeling_unimol import UniMolConfig, UniMolModel

class Head(nn.Module):
    def __init__(self, cell_dim=1280, repr_dim=512, fused_dim=512, target_dim=978):
        super().__init__()
        self.cell_proj = nn.Linear(cell_dim, fused_dim)  
        self.repr_proj = nn.Linear(repr_dim, fused_dim)  
        self.fc = nn.Linear(fused_dim, target_dim)       
        
        
        self.cell_weight = nn.Parameter(torch.tensor(0.5))
        self.repr_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, cell_repr, repr):
        
        cell_repr = self.cell_proj(cell_repr)  
        repr = self.repr_proj(repr)            
        fused_repr = self.cell_weight * cell_repr + self.repr_weight * repr

        return self.fc(fused_repr) 


class CureALLModule(LightningModule):

    def __init__(
        self,
        net_config: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        loss_config: Dict[str, Any],
        metric_config: Dict[str, Any],
        compile: bool,
        net: Optional[torch.nn.Module] = None,
    ) -> None:
        """Initialize a `CureALLModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.loss_config = loss_config
        if net is not None:
            self.net = net
        else:
            self.net_config = net_config
        model_type = self.net_config.model_type

        unimol_config = UniMolConfig(**self.net_config.unimol)
        self.mol_model = UniMolModel(unimol_config)
        _ckpt = torch.load(self.net_config.unimol_pretrained_weights)
        ckpt = convert_unimol_ckpt(_ckpt["model"], self.net_config.unimol.delta_pair_repr_norm_loss < 0)
        self.mol_model.load_state_dict(ckpt)

        uce_config = UCEConfig(**self.net_config.uce)
        self.cell_model = UCEModel(uce_config)

        if model_type == "unimol":
            self.cell_model.freeze()
        elif model_type == "uce":
            self.mol_model.freeze()
        elif model_type == "lp":
            self.mol_model.freeze()
            self.cell_model.freeze()
        else:
            print("Use default model: unimol plus uce")
        
        # default
        # self.uce_cast = nn.Linear(self.net_config.uce.output_dim, self.net_config.unimol.encoder_embed_dim)
        # self.concat = nn.Linear(self.net_config.unimol.encoder_embed_dim, self.net_config.target_dim)
        self.head = Head(cell_dim=1280,repr_dim=512, fused_dim=512, target_dim=978)
        # 1280 512
        # 512 978
        # [jyq]  (1280 + 512) 1792 -> 978  2770*2/3
        # self.head = nn.Sequential(
		# 	nn.Linear(self.net_config.unimol.encoder_embed_dim + self.net_config.uce.output_dim, 1850),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(1850, 1850),
		# 	nn.ReLU(inplace=True),
		# 	nn.Linear(1850, self.net_config.target_dim),
		# )

        self.uce_linear = nn.Linear(self.net_config.uce.output_dim, 489)
        self.smiles_linear = nn.Linear(self.net_config.unimol.encoder_embed_dim, 489)
        
        # loss function
        self.criterion = torch.nn.MSELoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_metric = CureALLMetrics()
        self.val_metric = CureALLMetrics()
        self.test_metric = CureALLMetrics()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # val max
        self.val_best = MaxMetric()
        

#     #### default
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Perform a forward pass through the model `self.net`.

#         :param x: A tensor of images.
#         :return: A tensor of logits.
#         """
#         if x.get("net_input") is not None:
#             x = x["net_input"]
#         repr, _ = self.mol_model(
#             src_tokens=x["src_tokens"],
#             src_coord=x["src_coord"],
#             src_distance=x["src_distance"],
#             src_edge_type=x["src_edge_type"],
#             features_only=True,
#         )

#         cell_repr = self.cell_model(x["uce_batches"][0], x["uce_batches"][1])
#         # get <cls> token representation
#         return self.concat(repr[:, 0, :] + self.uce_cast(cell_repr))
    
    #### [jyq]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        if x.get("net_input") is not None:
            x = x["net_input"]
        repr, _ = self.mol_model(
            src_tokens=x["src_tokens"],
            src_coord=x["src_coord"],
            src_distance=x["src_distance"],
            src_edge_type=x["src_edge_type"],
            features_only=True,
        )
        
        cell_repr = self.cell_model(x["uce_batches"][0], x["uce_batches"][1])
        preds = torch.cat([self.uce_linear(cell_repr), self.smiles_linear(repr[:, 0, :])], dim=-1)
        # get <cls> token representation
        # return self.head(torch.cat((repr[:, 0, :], cell_repr), dim=1))
        # smiles_repr = self.smiles_linear(repr[:, 0, :])
        # cell_repr = self.uce_linear(cell_repr)
        # preds = self.head(cell_repr, repr[:, 0, :])
        
        return preds, cell_repr, repr[:, 0, :]

    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.train_loss.reset()
        self.train_metric.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        item = batch
        preds, cell_embed, smiles_embed = self.forward(item["net_input"])
        target = item["label"]["target"]
        control = item["label"]["control"]

        def safe_normalize(x, eps=1e-9):
            return F.normalize(x, p=2, dim=1, eps=eps)

        preds = safe_normalize(preds)
        target = safe_normalize(target)
        control = safe_normalize(control)
        
        regression_loss = self.criterion(preds, target)
        control_loss = 1 - F.cosine_similarity(preds, control).mean()
        margin = 0.5
        positive_pairs = F.pairwise_distance(preds, target)
        negative_pairs = F.pairwise_distance(preds, control)
        contrastive_loss = F.relu(margin + positive_pairs - negative_pairs).mean()
        alpha, beta, gamma = 1.0, 0.1, 0.1
        loss = alpha * regression_loss + beta * control_loss + gamma * contrastive_loss
        # cont = target - control
        # eps = 1e-8
        # pstd = preds.std(dim=0)
        # cstd = cont.std(dim=0)
        # pstd = torch.where(pstd == 0, torch.tensor(eps, device=pstd.device), pstd)
        # cstd = torch.where(cstd == 0, torch.tensor(eps, device=cstd.device), cstd)
        # preds_normalized = (preds - preds.mean(dim=0)) / pstd # Normalize preds
        # cont_normalized = (cont - cont.mean(dim=0)) / cstd # Normalize contrastive targets
        # loss = self.criterion(preds, cont)

        # loss = (
        #     self.criterion(preds, target) * self.loss_config.alpha_mse
        #     + self.criterion(preds - control, target - control) * self.loss_config.alpha_mse_control
        #     + binary_top_k_loss(preds - control, target - control) * self.loss_config.alpha_topk
        # ) / 3
        # return loss, preds, item["label"]
        return loss, preds, cont

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, label = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)

        if batch_idx % self.hparams.metric_config.cal_every_n_batch == 0:
            # self.train_metric(preds, label["target"])
            self.train_metric(preds, label)

        self.log("train/loss", self.train_loss, on_step=True, prog_bar=True)
        # TODO: check if this is correct
        for k, v in self.train_metric.compute().items():
            self.log(f"train/{k}", v, on_step=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        for k, v in self.train_metric.compute().items():
            self.log(f"train_batch/{k}", v, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        # self.val_metric(preds, targets["target"])
        self.val_metric(preds, targets)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        self.val_best(self.val_metric.compute()["r2"])
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log("val/r2_best", self.val_best.compute(), sync_dist=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        for k, v in self.val_metric.compute().items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_metric.reset()

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_metric(preds, targets["target"])

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in self.test_metric.compute().items():
            self.log(f"test/{k}", v, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.test_loss.reset()
        self.test_metric.reset()

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def dynamic_weighted_loss(self, preds, target, control):
        mse_loss = self.criterion(preds, target)
        mse_control_loss = self.criterion(preds - control, target - control)
        topk_loss = binary_top_k_loss(preds - control, target - control)
        
        total_loss = (
            mse_loss * (1 / (mse_loss + 1e-5))
            + mse_control_loss * (1 / (mse_control_loss + 1e-5))
            + topk_loss * (1 / (topk_loss + 1e-5))
        )
        return total_loss
    
    def hierarchical_loss(self, preds, target, control, top_k=10):
        top_indices = torch.topk((target - control).abs(), top_k).indices
        top_preds = preds[:, top_indices]
        top_target = target[:, top_indices]
        
        high_importance_loss = self.criterion(top_preds, top_target)
        low_importance_loss = self.criterion(preds, target)
        
        return high_importance_loss * 0.7 + low_importance_loss * 0.3
    
    def entropy_loss(preds, control):
        p_pred = torch.softmax(preds - control, dim=-1)
        return -torch.sum(p_pred * torch.log(p_pred + 1e-5)) / p_pred.shape[0]



        
