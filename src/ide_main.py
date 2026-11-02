import argparse
import numpy as np

from data_utils import load_info, save_ide_info
from ide import ide_abid, ide_corrint, ide_danco, ide_lpca, ide_mle, ide_twonn
from knn import precompute_knn


AVAILABLE_MANIFOLDS = [
    'lin_nsp',
    'gman_proj',
    'gman_vec',
    'flag_vec',
    'steifel_proj',
    'steifel_vec1',
]

IDE_METHODS = [
    'lpca_maxgap',
    'lpca_ratio',
    'lpca_fo',
    'mle',
    'danco',
    'corrint',
    'twonn',
    'abid',
]


def parser():
    """Parse command-line arguments for IDE experiments."""
    parser = argparse.ArgumentParser(description='IDE estimation for supported manifolds.')

    parser.add_argument('--manifold_name', type=str, default='lin_nsp', help='Name of the manifold.')
    parser.add_argument('--d1', type=int, default=20, help='First integer dimension; -1 implies not needed.')
    parser.add_argument('--d2', type=int, default=10, help='Second integer dimension; -1 implies not needed.')
    parser.add_argument('--d3', type=int, default=-1, help='Third integer dimension; -1 implies not needed.')

    parser.add_argument('--ide_name', type=str, default='lpca_maxgap', help='IDE technique to run.')
    parser.add_argument('--k', type=int, default=100, help='k nearest neighbors')
    parser.add_argument('--N', type=int, default=5000, help='# of samples')
    parser.add_argument('--sigma', type=float, default=0.0, help='Noise scale for additive noise')
    parser.add_argument('--noise_type', type=str, default='isoGauss', help='Noise type for experiments.')

    parser.add_argument('--eps', type=float, default=0.1, help='epsilon parameter for lPCA')
    parser.add_argument('--dNoise', type=float, default=0.1, help='noise parameter for MLE')
    parser.add_argument('--dFrac', type=float, default=0.1, help='discard fraction parameter for TwoNN')
    parser.add_argument('--k1', type=int, default=10, help='k1 parameter for CorrInt/DANCo')
    parser.add_argument('--k2', type=float, default=2.0, help='Multiplier for CorrInt upper neighborhood size')

    parser.add_argument('--dg_seed', type=int, default=42, help='Seed for manifold data generation.')
    parser.add_argument('--seed', type=int, default=0, help='Seed to control randomness for IDE experiments.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of available CPUs.')

    return parser.parse_args()


def get_manifold_params(args):
    """Build the parameter tuple for the requested manifold."""
    if args.manifold_name == 'flag_vec':
        return (args.d1, args.d2, args.d3, args.dg_seed)
    return (args.d1, args.d2, args.dg_seed)


def main():
    """Load samples, run the requested IDE method, and save results."""
    args = parser()

    if args.manifold_name not in AVAILABLE_MANIFOLDS:
        raise ValueError(f'Please choose one of: {AVAILABLE_MANIFOLDS}')

    m_params = get_manifold_params(args)

    try:
        info_dict_m = load_info(
            params=m_params,
            manifold_name=args.manifold_name,
            root_dir='./data/samples/',
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError('Samples file not found.') from exc

    samples = info_dict_m['samples']
    if samples.ndim > 2:
        samples = samples.reshape((samples.shape[0], -1))

    if args.N > samples.shape[0]:
        raise ValueError(
            f'Requested N={args.N}, but only {samples.shape[0]} samples are available.'
        )

    if args.k >= args.N - 1:
        args.k = args.N - 2

    np.random.seed(args.seed)
    indices = np.random.permutation(samples.shape[0])[:args.N]
    samples = samples[indices]
    knn = precompute_knn(samples, kmax=args.k, n_jobs=args.n_jobs)
    knn = (knn[0][:, 1:], knn[1][:, 1:])

    if args.sigma > 0:
        da = samples.shape[1] 
        if args.noise_type=='isoGauss':
            mean = np.zeros(da)
            cov = np.eye(da)/da
            noise = np.random.multivariate_normal(mean=mean, cov=cov, size=samples.shape[0])
        elif args.noise_type=='exp':
            noise = np.random.exponential(scale=1.0, size=samples.shape)
            signs = np.random.choice([-1, 1], size=samples.shape)
            noise = signs*noise
        elif args.noise_type=='uniform':
            noise = np.random.uniform(low = -1.0,high=1.0,size=samples.shape)
        elif args.noise_type=='anisoGauss':
            mean = np.zeros(da)
            u = np.random.randn(da,da)/da
            cov = u@(u.T)
            cov = cov/np.trace(cov)
            noise = np.random.multivariate_normal(mean=mean, cov=cov, size=samples.shape[0])
        elif args.noise_type=='uncorrGauss':
            mean = np.zeros(da)
            cov = np.diag( np.random.rand(da) )*2.0/da
            cov = cov/np.trace(cov)
            noise = np.random.multivariate_normal(mean=mean, cov=cov, size=samples.shape[0])
        samples += args.sigma*noise

    if args.ide_name not in IDE_METHODS:
        raise ValueError(f'Please choose one of: {IDE_METHODS}')
    elif args.ide_name == 'lpca_maxgap':
        params = (args.k, None, args.seed)
        info_dict = ide_lpca(samples=samples, precomputed_knn=knn, manifold_name=args.manifold_name, m_params=m_params, ver='maxgap', params=params, n_jobs=args.n_jobs)
    elif args.ide_name == 'lpca_ratio':
        params = (args.k, args.eps, args.seed)
        info_dict = ide_lpca(samples=samples, precomputed_knn=knn, manifold_name=args.manifold_name, m_params=m_params, ver='ratio', params=params, n_jobs=args.n_jobs)
    elif args.ide_name == 'lpca_fo':
        params = (args.k, args.eps, args.seed)
        info_dict = ide_lpca(samples=samples, precomputed_knn=knn, manifold_name=args.manifold_name, m_params=m_params, ver='fo', params=params, n_jobs=args.n_jobs)
    elif args.ide_name == 'mle':
        params = (args.k, args.dNoise, args.seed)
        info_dict = ide_mle(samples=samples, precomputed_knn=knn, manifold_name=args.manifold_name, m_params=m_params, params=params, n_jobs=args.n_jobs)
    elif args.ide_name == 'danco':
        if args.k1 <= args.k:
            args.k1 = args.k
        params = (args.k, args.k1, args.seed)
        info_dict = ide_danco(samples=samples, precomputed_knn=knn, manifold_name=args.manifold_name, m_params=m_params, params=params, n_jobs=args.n_jobs)
    elif args.ide_name == 'twonn':
        params = (args.k, args.dFrac, args.seed)
        info_dict = ide_twonn(samples=samples, precomputed_knn=knn, manifold_name=args.manifold_name, m_params=m_params, params=params, n_jobs=args.n_jobs)
    elif args.ide_name == 'corrint':
        k2 = int(args.k2 * args.k1)
        if k2 >= args.N - 1:
            k2 = args.N - 2
            args.k1 = k2 // 2
        params = (args.k1, k2, args.seed)
        info_dict = ide_corrint(samples=samples, precomputed_knn=knn, manifold_name=args.manifold_name, m_params=m_params, params=params, n_jobs=args.n_jobs)
    else:
        params = (args.k, args.seed)
        info_dict = ide_abid(samples=samples, precomputed_knn=knn, manifold_name=args.manifold_name, m_params=m_params, params=params, n_jobs=args.n_jobs)

    params += (args.N, args.sigma)
    save_ide_info(
        info_dict=info_dict,
        ide_name=args.ide_name,
        params=params,
        manifold_name=args.manifold_name,
        m_params=m_params,
        root_dir='./data/ide/',
    )


if __name__ == '__main__':
    main()
