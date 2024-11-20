import torch
from torchmetrics import Metric
from torchmetrics.functional.regression import pearson_corrcoef, r2_score, spearman_corrcoef


class CureALLMetrics(Metric):
    def __init__(self, simple: bool = False):
        super().__init__()
        self.simple = simple
        self.add_state("r2", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("spear", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("pear", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds, target = self._input_format(preds, target)
        for pred, tgt in zip(preds, target):
            if self.simple:
                self.spear += spearman_corrcoef(pred, tgt)
            else:
                self.r2 += r2_score(pred, tgt)
                self.spear += spearman_corrcoef(pred, tgt)
                self.pear += pearson_corrcoef(pred, tgt)
            self.total += 1
        pass

    def compute(self):
        return (
            {"spear": self.spear / self.total}
            if self.simple
            else {"r2": self.r2 / self.total, "spear": self.spear / self.total, "pear": self.pear / self.total}
        )

    def _input_format(self, preds, target):
        """
        Ensure that preds and target are in the correct format.
        """
        return preds, target
