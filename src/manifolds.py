import math
from itertools import combinations

import numpy as np
from scipy.linalg import svd
from scipy.stats import ortho_group

MAX_SAMPLE = 10_000


def random_Grassmannian_vector(n, k, seed=42):
    """Sample a Grassmannian embedding vector."""
    np.random.seed(seed)
    vector = np.zeros(math.comb(n, k))
    m = ortho_group.rvs(dim=n)

    for i, c in enumerate(combinations(range(n), k)):
        vector[i] = np.linalg.det(m[c, :k])

    return vector


def random_Flag_vector(n, k1, k2, seed=42):
    """Sample a flag-manifold embedding vector."""
    np.random.seed(seed)
    vector = np.zeros((math.comb(n, k1), math.comb(n, k2)))
    m = ortho_group.rvs(dim=n)

    for i, c1 in enumerate(combinations(range(n), k1)):
        for j, c2 in enumerate(combinations(range(n), k2)):
            vector[i, j] = np.linalg.det(m[c1, :k1]) * np.linalg.det(m[c2, :k2])

    return vector.reshape(-1)


def data_gen_lin_nsp(N=MAX_SAMPLE, params=(200, 100, 42)):
    """Generate samples from a linear null space.

    params: (d1, d2, seed)
    """
    d1, d2, seed = params
    np.random.seed(seed)

    U = np.random.randn(d1, d2)
    V = np.random.randn(d2, d1)
    A = np.dot(U, V)

    _, _, Vt = svd(A)
    null_space_basis = Vt[d2:].T

    null_space_dim = d1 - d2
    coefficients = np.random.randn(N, null_space_dim)
    null_space_vectors = np.dot(coefficients, null_space_basis.T)

    return {
        'samples': null_space_vectors,
        'params': params,
        'matrixA': A,
    }


def data_gen_gman_proj(N=MAX_SAMPLE, params=(6, 3, 42)):
    """Generate Grassmannian projector samples.

    params: (n, k, seed)
    """
    d1, d2, seed = params
    np.random.seed(seed)

    projectors = np.zeros((N, d1, d1))
    D = np.zeros((d1, d1))
    np.fill_diagonal(D[:d2, :d2], 1)
    list_of_matrices = []

    for i in range(N):
        random_matrix = np.random.randn(d1, d1)
        Q, _ = np.linalg.qr(random_matrix)
        list_of_matrices.append(Q)
        projectors[i] = Q @ D @ Q.T

    return {
        'samples': projectors,
        'params': params,
        'listQs': np.array(list_of_matrices),
    }


def data_gen_gman_vec(N=MAX_SAMPLE, params=(6, 3, 42)):
    """Generate Grassmannian vector-embedding samples.

    params: (n, k, seed)
    """
    d1, d2, seed = params
    np.random.seed(seed)
    dim = math.comb(d1, d2)
    vectors = np.zeros((N, dim))

    for i in range(N):
        vectors[i, :] = random_Grassmannian_vector(n=d1, k=d2, seed=seed + i)

    return {
        'samples': vectors,
        'params': params,
    }


def data_gen_steifel_proj(N=MAX_SAMPLE, params=(6, 3, 42)):
    """Generate Stiefel projector samples.

    params: (n, k, seed)
    """
    d1, d2, seed = params
    np.random.seed(seed)
    projectors = np.zeros((N, d1, d2))

    for i in range(N):
        random_matrix = np.random.randn(d1, d1)
        Q, _ = np.linalg.qr(random_matrix)
        projectors[i] = Q[:, :d2]

    return {
        'samples': projectors,
        'params': params,
    }


def data_gen_steifel_vec1(N=MAX_SAMPLE, params=(10, 5, 42)):
    """Generate Stiefel vector-embedding samples.

    params: (n, k, seed)
    """
    d1, d2, seed = params
    np.random.seed(seed)

    dim = math.comb(d1, d2)
    vectors = np.zeros((N, dim + d2**2))
    for i in range(N):
        vectors[i, :dim] = random_Grassmannian_vector(n=d1, k=d2, seed=seed + i)
        m = np.reshape(ortho_group.rvs(dim=d2), -1)
        vectors[i, dim:] = m

    return {
        'samples': vectors,
        'params': params,
    }


def data_gen_flag_vec(N=MAX_SAMPLE, params=(4, 3, 2, 42)):
    """Generate flag-manifold vector samples.

    params: (n, k1, k2, seed)
    """
    d1, d2, d3, seed = params
    np.random.seed(seed)
    dim = math.comb(d1, d2) * math.comb(d1, d3)
    vectors = np.zeros((N, dim))

    for i in range(N):
        vectors[i, :] = random_Flag_vector(n=d1, k1=d2, k2=d3, seed=seed + i)

    return {
        'samples': vectors,
        'params': params,
    }
