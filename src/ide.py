import numpy as np
import os
from sklearn.decomposition import PCA
from scipy.stats import mode
from skdim.id import lPCA, MLE, TwoNN, DANCo, CorrInt, ESS, MADA
from sklearn.decomposition import PCA
from knn import *

# Helper functions #

def exvar_pca(samples, precomputed_knn,
                manifold_name = 'lin_nsp',
                m_params=(100,42),
                params=(100,0),
                root_dir = './data/ide/,',
                savefile=True):
    '''
    Compute explained variance ratios either globally or pointwise.

    Returns the explained_variance_ratios.
    '''
    k, seed = params
    N, amb_dim = samples.shape

    if k >= N:
        k = -1
    
    np.random.seed(seed)

    if k == -1:
        evar = PCA().fit(samples).explained_variance_ratio_
        evar_pw = np.reshape(evar, (evar.shape[0],1) )
    else:
        indices = precomputed_knn[1][:,:k]
        evar_pw = []

        for i in range(N):
            local_nbd = samples[indices[i]]
            evar = PCA().fit(local_nbd).explained_variance_ratio_
            evar_pw.append(evar)
        
        evar_pw = np.array(evar_pw)
    
    if savefile:
        filename = f'{root_dir}_{manifold_name}_p_{m_params}_pca_p_{params}_exvar.txt'
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savetxt(filename, evar_pw)

    return evar_pw


def abids(X, knns, raw = False, n_jobs=1):
    '''
    Compute pointwise ABID estimates from precomputed neighborhoods.
    Code credit : Erik Thordsen and Erich Schubert.
    '''
    N, dim = X.shape
    dists, indices = knns
    lides = []
    for i in range(N):
        neighbors = X[indices[i]]-X[i]

        neighbor_norms = np.linalg.norm(neighbors,axis=1)
        neighbors = neighbors[neighbor_norms > 0] / neighbor_norms[neighbor_norms > 0][:,None]

        k = neighbors.shape[0]
        if neighbors.shape[0] <= neighbors.shape[1]:
            cosines = neighbors.dot(neighbors.T)
            ssq_eigvals = np.sum(np.square(cosines))
        else:
            mat = neighbors.T.dot(neighbors)
            ssq_eigvals = np.sum(np.square(mat))
        if raw:
            lid =(k**2-k)/(ssq_eigvals-k)
        else:
            lid = k**2 / ssq_eigvals
        lides.append(lid)
    lides = np.array(lides)

    return lides

### Helper functions end ###


### lPCA ###

def ide_lpca(samples=None, precomputed_knn=None,
            manifold_name = 'lin_nsp',
            m_params=(100,42), 
            ver = 'maxgap', params = (100,0.1,0),
            n_jobs=1, root_dir = '.data/ide_data/'):
    '''
    Run lPCA-based pointwise intrinsic-dimension estimation.

    Params : (k: int, eps: float or None, seed: int)
    '''
    k, eps, seed = params
    exvar_params = (k, seed)
    
    filename = f'{root_dir}_{manifold_name}_p_{m_params}_pca_p_{exvar_params}_exvar.txt'
    
    if os.path.exists(filename):
        evar_pw =np.loadtxt(filename)
        N = evar_pw.shape[0]
    else:
        assert samples is not None, 'File does not exist, need samples to proceed!'
        evar_pw = exvar_pca(samples, precomputed_knn, manifold_name, m_params=m_params, params=(k,seed))
    
    if samples is not None:
        N,amb_dim = samples.shape
    
    if ver == 'maxgap':
        gaps = evar_pw[:,:-1]/(evar_pw[:,1:] + 1e-6)
        lides = np.nanargmax(gaps, axis=1) + 1
    elif ver == 'ratio':
        cum_ratio = np.cumsum(evar_pw,axis=1)/np.sum(evar_pw,axis=1,keepdims=True)
        lides = np.zeros((N))
        for i in range(N):
            lides[i] = np.searchsorted(cum_ratio[i,:], 1-eps, side='right')+1
    elif ver == 'fo':
        lides = np.zeros((N))
        for i in range(N):
            meig = np.max(evar_pw[i,:])*(1-eps)
            lides[i] = np.searchsorted(evar_pw[i,:], meig, side='right')+1
    else:
        available_ver = ['maxgap', 'ratio', 'fo' ]
        raise ValueError(f'{ver} not implemented. Available methods are {available_ver}')

    info_dict = {'lides' : lides,
                 'params' : params,
                 'ver' : ver,
                 'manifold': manifold_name,
                 'm_params':m_params,
                 'exp_var_ratios' : evar_pw}
    
    return info_dict


def ide_mle(samples=None, precomputed_knn=None,
             manifold_name = 'lin_nsp',
             m_params=(100,42), 
             params = (100,0.0,0),
             n_jobs=1):
    '''
    Maximum Likelihood Estimation.
    Params: (k: int, sigma: float, seed: int)
    '''

    k, sigma, seed = params
    estimator = MLE(sigma=sigma)
    lides = estimator.fit(X=samples, 
                      precomputed_knn_arrays=precomputed_knn,
                      n_jobs=n_jobs).dimension_pw_
    
    info_dict = {'lides':lides,
                 'params': params,
                 'manifold': manifold_name,
                 'm_params':m_params,}

    return info_dict


def ide_danco(samples=None, precomputed_knn=None,
             manifold_name = 'lin_nsp',
             m_params=(100,42), 
             params = (5,7,0),
             n_jobs=1):
    '''
    Distance and Angle-Norm concentration.
    Params: (k : int, k1 : int, seed: int)
    Ensure k1 > k+1.
    '''
    k, k1, seed = params

    estimator = DANCo(k=k-2)
    lides = estimator.fit_pw(X=samples, 
                      n_neighbors=precomputed_knn[1].shape[0]-2,
                      n_jobs=n_jobs).dimension_pw_
    
    info_dict = {'lides':lides,
                 'params': params,
                 'manifold': manifold_name,
                 'm_params':m_params,}

    return info_dict


def ide_twonn(samples=None, precomputed_knn=None,
             manifold_name = 'normal',
             m_params=(100,42), 
             params = (100,0.1,0),
             n_jobs=1):
    '''
    Two Nearest Neighbors estimation.
    Params: (k: int, dFrac : float, seed: int)
    '''
    k, dFrac, seed = params

    estimator = TwoNN(discard_fraction=dFrac)
    lides = estimator.fit_pw(X=samples, 
                      precomputed_knn=precomputed_knn[1],
                      n_neighbors=precomputed_knn[1].shape[0]-2,
                      n_jobs=n_jobs).dimension_pw_
    
    info_dict = {'lides':lides,
                 'params': params,
                 'manifold': manifold_name,
                 'm_params':m_params,}

    return info_dict


def ide_corrint(samples=None, precomputed_knn=None,
             manifold_name = 'normal',
             m_params=(100,42), 
             params = (10,20,0),
             n_jobs=1):
    '''
    Correlation integral estimation.
    Params: (k1, k2, seed)
    '''
    k1, k2, seed = params

    estimator = CorrInt(k1=k1,k2=k2)
    lides = estimator.fit_pw(X=samples, 
                      n_neighbors=precomputed_knn[1].shape[0]-2,
                      n_jobs=n_jobs).dimension_pw_
    
    info_dict = {'lides':lides,
                 'params': params,
                 'manifold': manifold_name,
                 'm_params':m_params,}

    return info_dict


def ide_abid(samples=None, precomputed_knn=None,
             manifold_name = 'normal',
             m_params=(100,42), 
             params = (20,0),
             n_jobs=1):
    '''
    Angle-based intrinsic dimension estimation.
    Params: (k, seed)
    '''
    k1, seed = params

    lides = abids(X=samples, 
                knns=precomputed_knn,
                n_jobs=n_jobs)
    
    info_dict = {'lides':lides,
                 'params': params,
                 'manifold': manifold_name,
                 'm_params':m_params,}

    return info_dict
