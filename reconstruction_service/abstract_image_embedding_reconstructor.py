from abc import ABC, abstractmethod
import numpy as np

class AbstractImageReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, embeddings: np.ndarray) -> np.ndarray:
        pass