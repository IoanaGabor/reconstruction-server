import numpy as np
import pickle

from reconstruction_service.abstract_voxel_embedding_extractor import AbstractVoxelEmbeddingExtractor
from sklearn.linear_model import Ridge
import numpy as np

class RidgeRegressionVoxelEmbeddingExtractor(AbstractVoxelEmbeddingExtractor):
    
    def extract(self, voxel_data, person_id: int) -> np.ndarray:
        weights_path = 'regressors/subj{:02d}/nsd_clipvision_regression_weights.npy'.format(person_id)
        bias_path = 'regressors/subj{:02d}/nsd_clipvision_regression_bias.npy'.format(person_id)
        
        means_train_path = 'statistics/subj{:02d}/clipvision_statistics_train_means.npy'.format(person_id)
        stds_train_path='statistics/subj{:02d}/clipvision_statistics_train_stds.npy'.format(person_id)
        means_test_path = 'statistics/subj{:02d}/clipvision_statistics_test_means.npy'.format(person_id)
        stds_test_path='statistics/subj{:02d}/clipvision_statistics_test_stds.npy'.format(person_id)
        norm_mean_train='statistics/subj{:02d}/norm_mean_train.npy'.format(person_id)
        norm_scale_train='statistics/subj{:02d}/norm_scale_train.npy'.format(person_id)
        mean_train=np.load(norm_mean_train)
        scale_train=np.load(norm_scale_train)

        #mean_train=-3.1423596606535043e-16
        #scale_train=0.9999435586284325
        #print(voxel_data)
        voxel_data=np.array(voxel_data, dtype=np.float64)
        np.set_printoptions(precision=16)
        #print(voxel_data)
        voxel_data=voxel_data/300
        voxel_data=(voxel_data - mean_train)/scale_train
        #print(voxel_data)
        #with open(weights_path, "rb") as f:
        #    reg_weights = pickle.load(f)
        #with open(statistics_path, "rb") as f:
        #    reg_statistics = pickle.load(f)
        
        reg_w = np.load(weights_path)#.astype(np.float64)
        print(reg_w)
        reg_b = np.load(bias_path)#.astype(np.float64)
        means_train = np.load(means_train_path)#.astype(np.float64)
        stds_train = np.load(stds_train_path)#.astype(np.float64)
        means_test = np.load(means_test_path)#.astype(np.float64)
        stds_test = np.load(stds_test_path)#.astype(np.float64)
        #print(stds_train)
        #print(stds_test)
        #print(len(means))
        #print(len(stds))
        num_embed, num_dim, num_voxels = reg_w.shape
        embeddings = np.zeros((num_embed, num_dim)).astype(np.float64)
        
        for i in range(num_embed):
            ridge = Ridge(alpha=60000, max_iter=50000, fit_intercept=True)
            ridge.coef_ = reg_w[i]
            ridge.intercept_ = reg_b[i]
            pred = ridge.predict(voxel_data.reshape(1, -1))
            #print(means_test[i])
            #print(stds_test[i])
            if stds_test[i].all() != 0:
                std_norm_test_latent = (pred - means_test[i]) / stds_test[i]
                embeddings[i] = std_norm_test_latent * stds_train[i] + means_train[i]
            else:
                embeddings[i] = pred
        #print(embeddings.tolist())
        #np.save("temp", embeddings)
        #embds = np.load("nsd_clipvision_predtest_nsdgeneral.npy")

        return embeddings
