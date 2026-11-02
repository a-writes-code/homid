import os
from sklearn.neighbors import NearestNeighbors

from data_utils import *

'''
Uses sklearn's NearestNeighbors() function to precompute 
kNNs for different functions. kNNs calculated for some 
kmax = 500. 
'''

KMAX = 600

def precompute_knn(samples,
                   kmax = KMAX, n_jobs = 1, 
                   return_tuple=True):
    '''
    Precompute nearest-neighbor distances and indices.
    '''
    nbr = NearestNeighbors(n_neighbors=kmax,algorithm='ball_tree',n_jobs=n_jobs).fit(samples)
    knn = nbr.kneighbors(samples)
    
    return knn if return_tuple else None


def check_knn_exists(manifold_name = 'lin_nsp',
                     params = (100,50,42),
                     root_dir = './data/knn/',
                     kmax = KMAX):
    '''
    Load saved kNN data if it exists for the requested manifold and parameters.
    '''
    FILENAME = f'{root_dir}_{manifold_name}_p_{params}_knn_{kmax}.h5'
    if os.path.exists(FILENAME):
        knn = load_knn_info(params=params, kmax=kmax,
                            manifold_name=manifold_name,
                            root_dir=root_dir)
    else:
        raise FileNotFoundError('Please run precompute_knn() for this sample first!')
        
    return knn
