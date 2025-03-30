from abc import ABC, abstractmethod
import numpy as np

class AbstractVoxelEmbeddingExtractor(ABC):
    @abstractmethod
    def extract(self, voxel_data: np.ndarray, person_id: int) -> np.ndarray:
        pass
