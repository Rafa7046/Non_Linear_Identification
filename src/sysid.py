import numpy as np
from utils import generate_combinations


def data_matrix(u, y, nu=1, ny=1):
    """
    Computes the data matrix Ψ , given the input and output vector.

    Parameters
    ----------
    u : array_like
        Input vector.
    y : array_like
        Output vector.
    nu : int, optional
        Number of inputs. The default is 1.
    ny : int, optional
        Number of outputs. The default is 1.

    Returns
    -------
    Ψ : array_like
        Data matrix consisting of nu + ny columns.
    """
    N = len(u)
    n = max(nu, ny)
    U = np.zeros((N-n, nu))
    Y = np.zeros((N-n, ny))

    for i in range(nu):
        U[:, -(i+1)] = u[i+1:N-n+i+1]

    for i in range(ny):
        Y[:, -(i+1)] = y[i+1:N-n+i+1]

    return np.hstack((Y, U))


def get_model_term(idxs, nu, ny):
    """
    Returns the model term corresponding to the given indices.

    Parameters
    ----------
    idxs : array_like
        Column indices of the data matrix.
    nu : int
        Number of inputs.
    ny : int
        Number of outputs.

    Returns
    -------
    str
        Model term corresponding to the given indices.
    """
    ans = ""
    for i in idxs:
        if i + 1 > ny:
            ans += f" u[k-{nu+ny-i}]"
        else:
            ans += f" y[k-{ny-i}]"
    return ans.strip()


def candidate_matrix(dm, nl):
    """
    Returns the candidate matrix with all possible combinations of columns
    with degree of non linearity given by nl.

    Parameters
    ----------

    dm : array_like
        Data matrix.
    nl : int
        Degree of non linearity.

    Returns
    -------
    array_like
        Candidate matrix.
    list
        List of all possible combinations of columns.
    """

    combinations = generate_combinations(list(range(dm.shape[1])), nl)

    cm = []
    for comb in combinations:
        m = dm[:, comb]
        product = np.prod(m, axis=1)
        cm.append(product)

    return np.column_stack(cm), combinations
