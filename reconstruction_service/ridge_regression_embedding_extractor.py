import numpy as np
import pickle

from reconstruction_service.abstract_voxel_embedding_extractor import AbstractVoxelEmbeddingExtractor
import numpy as np

class RidgeRegressionVoxelEmbeddingExtractor(AbstractVoxelEmbeddingExtractor):
    
    def extract(self, voxel_data: np.ndarray, person_id: int) -> np.ndarray:
        weights_path = 'data/regression_weights/subj{:02d}/clipvision_regression_weights.pkl'.format(person_id)
        with open(weights_path, "rb") as f:
            reg_weights = pickle.load(f)
        
        reg_w = reg_weights['weight']  
        reg_b = reg_weights['bias']    
        
        num_embed, num_dim, num_voxels = reg_w.shape
        embeddings = np.zeros((num_embed, num_dim))
        
        for i in range(num_embed):
            embeddings[i] = np.dot(voxel_data, reg_w[i].T) + reg_b[i]
        
        return embeddings