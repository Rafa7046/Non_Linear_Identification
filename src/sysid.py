import numpy as np
import pandas as pd


def data_matrix(u, y, nu=1, ny=1):
    """ Given the input and output data, return the data matrix with nu and ny delays"""
    N = len(u)
    U = np.zeros((N-nu, nu))
    Y = np.zeros((N-ny, ny))
    
    for i in range(1, nu+1):
        U[:, i] = u[i:N-nu+i]
    
    for i in range(1, ny+1):
        Y[:, i] = y[i:N-ny+i]
    return np.hstack((Y, U))