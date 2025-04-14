import torch
from ssim_metric import SSIMMetric
from pixcorr_metric import PixCorrMetric

class MetricsCalculator:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.metrics = [SSIMMetric(), PixCorrMetric()]

    def compute_all(self, img1: torch.Tensor, img2: torch.Tensor):
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        results = {
            metric.name: metric.compute(img1, img2)
            for metric in self.metrics
        }

        return results