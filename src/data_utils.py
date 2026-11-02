import numpy as np
import h5py
import pickle
import os

KMAX = 600


def _ensure_parent_dir(filename):
    """Create the parent directory for a file path when needed."""
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)


def save_info(info_dict=None, params = (100,50,42),
              manifold_name='lin_nsp', 
              root_dir = './data/samples/'):
    """Save generated manifold samples and metadata to an HDF5 file.

    Args:
        info_dict (dict): Dictionary containing arrays and metadata.
        params (tuple): Parameter tuple used to generate the data.
        manifold_name (str): Name of the manifold.
        root_dir (str): Output file prefix.
    """
    FILENAME = f'{root_dir}_{manifold_name}_p_{params}.h5'
    _ensure_parent_dir(FILENAME)
    with h5py.File(FILENAME, "w") as f:
        for key, val in info_dict.items():
            if isinstance(val, np.ndarray):
                f.create_dataset(key, data=val, compression="lzf")

        if "params" in info_dict and not isinstance(info_dict["params"], np.ndarray):
            f.attrs["params"] = np.void(pickle.dumps(info_dict["params"]))


def load_info(params = (100,50,42),
              manifold_name='lin_nsp', 
              root_dir = './data/samples/'):
    """Load generated manifold samples and metadata from an HDF5 file."""
    FILENAME = f'{root_dir}_{manifold_name}_p_{params}.h5'
    if os.path.exists(FILENAME):
        with h5py.File(FILENAME, "r") as f:
            info_dict = {key: f[key][()] for key in f.keys()}
            if "params" in f.attrs:
                info_dict["params"] = pickle.loads(f.attrs["params"].tobytes())
        return info_dict
    else:
        raise FileNotFoundError(f"File not found: {FILENAME}")


def save_knn_info(knn=None, 
                  params=(100,50,42),
                  kmax=KMAX,
                  manifold_name='lin_nsp', 
                  root_dir = './data/knn/'):
    """Save precomputed nearest-neighbor distances and indices to HDF5."""
    FILENAME = f'{root_dir}_{manifold_name}_p_{params}_knn_{kmax}.h5'
    _ensure_parent_dir(FILENAME)
    with h5py.File(FILENAME, "w") as f:
        grp = f.create_group("knn")
        grp.create_dataset("dist", data=knn[0], compression="lzf")
        grp.create_dataset("indices", data=knn[1], compression="lzf")


def load_knn_info(params=(100,50,42),
                  kmax=KMAX,
                  manifold_name='lin_nsp', 
                  root_dir = './data/knn/'):
    """Load precomputed nearest-neighbor distances and indices from HDF5."""
    FILENAME = f'{root_dir}_{manifold_name}_p_{params}_knn_{kmax}.h5'
    
    if not os.path.exists(FILENAME):
        raise FileNotFoundError(f"File not found: {FILENAME}")
    
    with h5py.File(FILENAME, "r") as f:
        knn_group = f["knn"]
        dists = knn_group["dist"][()]
        indices = knn_group["indices"][()]
        knn = (dists, indices)
        
    return knn


def save_ide_info(info_dict=None, 
                  ide_name='lpca_maxgap',
                  params=(100,0.1,0),
                  manifold_name='lin_nsp',
                  m_params=(100,30,0),
                  root_dir = './data/ide_data/'):
    """Save intrinsic-dimension estimation results to a pickle file."""
    FILENAME = f'{root_dir}_{ide_name}_p_{params}_{manifold_name}_mp_{m_params}.pkl'
    _ensure_parent_dir(FILENAME)

    with open(FILENAME, 'wb') as fp:
        pickle.dump(info_dict, fp)


def load_ide_info(ide_name='lpca_maxgap',
                  params=(100,0.1,0),
                  manifold_name='lin_nsp',
                  m_params=(100,30,0),
                  root_dir = './data/ide_data/'):
    """Load intrinsic-dimension estimation results from a pickle file."""
    FILENAME = f'{root_dir}_{ide_name}_p_{params}_{manifold_name}_mp_{m_params}.pkl'

    if os.path.exists(FILENAME):
        with open(FILENAME, 'rb') as fp:
            info_dict = pickle.load(fp)
    else:
        raise FileNotFoundError(f"File not found: {FILENAME}")
    
    return info_dict
