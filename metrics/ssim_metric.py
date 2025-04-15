import torch
import numpy as np
from torchvision import transforms
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from metrics.image_similarity_metric import ImageSimilarityMetric

class SSIMMetric(ImageSimilarityMetric):
    def __init__(self):
        self.name="SSIM"
        self.preprocess = transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR)

    def compute(self, original: torch.Tensor, reconstructed: torch.Tensor) -> float:
        original = self.preprocess(original).permute(0,2,3,1).cpu().numpy()
        reconstructed = self.preprocess(reconstructed).permute(0,2,3,1).cpu().numpy()
        original_gray = rgb2gray(original)
        reconstructed_gray = rgb2gray(reconstructed)
        return float(ssim(
            reconstructed_gray, original_gray,
            multichannel=True,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0
        ))
