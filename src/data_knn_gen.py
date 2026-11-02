import argparse

from manifolds import (
    data_gen_flag_vec,
    data_gen_gman_proj,
    data_gen_gman_vec,
    data_gen_lin_nsp,
    data_gen_steifel_proj,
    data_gen_steifel_vec1,
)
from data_utils import save_info


AVAILABLE_MANIFOLDS = [
    'lin_nsp',
    'gman_proj',
    'gman_vec',
    'flag_vec',
    'steifel_proj',
    'steifel_vec1',
]


def parser():
    """Parse command-line arguments for manifold sample generation.

    Returns:
        argparse.Namespace: Parsed arguments controlling the manifold type,
        manifold parameters, and random seed.
    """
    parser = argparse.ArgumentParser(description='Create manifold samples.')

    parser.add_argument(
        '--manifold_name',
        type=str,
        default='lin_nsp',
        help='Name of the manifold. Must match one of AVAILABLE_MANIFOLDS.',
    )
    parser.add_argument('--d1', type=int, default=20, help='First integer dimension; -1 implies not needed.')
    parser.add_argument('--d2', type=int, default=10, help='Second integer dimension; -1 implies not needed.')
    parser.add_argument('--d3', type=int, default=-1, help='Third integer dimension; -1 implies not needed.')
    parser.add_argument('--seed', type=int, default=42, help='Seed to control randomness.')

    return parser.parse_args()


GENERATOR_MAP = {
    'lin_nsp': lambda args: ((args.d1, args.d2, args.seed), data_gen_lin_nsp(params=(args.d1, args.d2, args.seed))),
    'gman_proj': lambda args: ((args.d1, args.d2, args.seed), data_gen_gman_proj(params=(args.d1, args.d2, args.seed))),
    'gman_vec': lambda args: ((args.d1, args.d2, args.seed), data_gen_gman_vec(params=(args.d1, args.d2, args.seed))),
    'flag_vec': lambda args: ((args.d1, args.d2, args.d3, args.seed), data_gen_flag_vec(params=(args.d1, args.d2, args.d3, args.seed))),
    'steifel_proj': lambda args: ((args.d1, args.d2, args.seed), data_gen_steifel_proj(params=(args.d1, args.d2, args.seed))),
    'steifel_vec1': lambda args: ((args.d1, args.d2, args.seed), data_gen_steifel_vec1(params=(args.d1, args.d2, args.seed))),
}


def main():
    """Generate samples for the requested manifold and save them to disk."""
    args = parser()

    if args.manifold_name not in AVAILABLE_MANIFOLDS:
        raise ValueError(f'Please choose one of: {AVAILABLE_MANIFOLDS}')

    params, info_dict = GENERATOR_MAP[args.manifold_name](args)

    save_info(
        info_dict=info_dict,
        params=params,
        manifold_name=args.manifold_name,
        root_dir='./data/samples/',
    )


if __name__ == '__main__':
    main()
