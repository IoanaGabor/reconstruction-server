import numpy as np
from torchvision import transforms
from metrics.image_similarity_metric import ImageSimilarityMetric

class PixCorrMetric(ImageSimilarityMetric):
    def __init__(self):
        self.name = "PixCorr"
        self.preprocess = transforms.Compose([
            transforms.Resize(425, interpolation=transforms.InterpolationMode.BILINEAR),
        ])

    def compute(self, original, reconstructed) -> float:
        original = self.preprocess(original).reshape(1, -1).cpu().numpy()
        reconstructed = self.preprocess(reconstructed).reshape(1, -1).cpu().numpy()
        return float(np.corrcoef(original[0], reconstructed[0])[0, 1])
