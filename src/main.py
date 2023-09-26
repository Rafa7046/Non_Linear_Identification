import numpy as np
import pandas as pd
import sysid as sd
import matplotlib.pyplot as plt

from utils import plot_io, plot_y


def gram_schmdit(cm, y, n):
    w1i = [cm[:, i] for i in range(cm.shape[1])]
    M = len(w1i)
    g1i = [w1i[i] @ y / (w1i[i] @ w1i[i]) for i in range(len(w1i))]
    ERRi = [g1i[i]**2 * w1i[i] @ w1i[i] / (y @ y) for i in range(len(w1i))]
    selected = [np.argmax(ERRi)]
    W = [w1i[selected[0]]]

    for k in range(1, n):
        ERRi = []
        for i in range(M):
            if i not in selected:
                alpha = [(W[j] @ w1i[i]) / (W[j] @ W[j]) for j in range(k)]
                wki = w1i[i] - sum([alpha[j] * W[j] for j in range(k)])
                gki = wki @ y / (wki @ wki)
                ERRi.append(gki**2 * wki @ wki / (y @ y))
            else:
                ERRi.append(0)
        selected.append(np.argmax(ERRi))
        W.append(w1i[selected[k]])

    return selected


if __name__ == "__main__":

    # loading the data

    df = pd.read_csv("data/ball-and-beam.csv")

    u = df["u"].values
    y = df["y"].values

    plot_io(
        u=u, y=y, title="Robot arm system"
    )

    # separate the data into training and testing sets

    n_train = int(0.2 * df.shape[0])
    u_train, u_test = u[-n_train:], u[:-n_train]
    y_train, y_test = y[-n_train:], y[:-n_train]

    # creating the data matrix for the train set

    nu, ny = 2, 2
    dm = sd.data_matrix(u=u_train, y=y_train, nu=nu, ny=ny)

    # creating the candidate matrix for the train set

    cm, comb = sd.candidate_matrix(dm, 3)

    Y = y_train[:-max(nu, ny)]

    # structure selection with gram_schmdit
    selected = gram_schmdit(cm, Y, 4)

    # parameter estimation on train set
    P = cm[:, selected]

    T = np.linalg.inv(P.T @ P) @ P.T @ Y

    y_pred = P @ T

    plot_y(
        y=Y, y_pred=y_pred, title="Robot arm system - Train set"
    )

    print("Selected terms:")
    for i in selected:
        print(sd.get_model_term(comb[i], nu, ny))

    # parameter estimation on test set

    dm = sd.data_matrix(u=u_test, y=y_test, nu=nu, ny=ny)

    # creating the candidate matrix for the test set

    cm, comb = sd.candidate_matrix(dm, 3)

    Y = y_test[:-max(nu, ny)]

    # parameter estimation on train set
    P = cm[:, selected]

    y_pred = P @ T

    plot_y(
        y=Y, y_pred=y_pred, title="Robot arm system - Test set"
    )

    # computes mean squared error

    mse = np.mean((y_pred - Y)**2)
    print("MSE=%.4f" % (mse))

    plt.show()
