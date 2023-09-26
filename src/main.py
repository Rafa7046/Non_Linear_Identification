import numpy as np
import pandas as pd
import sysid as sd
import matplotlib.pyplot as plt

from utils import plot_io, plot_y


def frols(cm, y, tol=0.05):
    w1i = [cm[:, i] for i in range(cm.shape[1])]
    M = len(w1i)
    g1i = [w1i[i] @ y / (w1i[i] @ w1i[i]) for i in range(len(w1i))]
    ERRi = [g1i[i]**2 * w1i[i] @ w1i[i] / (y @ y) for i in range(len(w1i))]
    selected = [np.argmax(ERRi)]
    W = [w1i[selected[0]]]

    selected_erri = [ERRi[selected[0]]]

    k = 1
    while 1 - np.sum(selected_erri) < tol:
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
        selected_erri.append(ERRi[selected[k]])
        k += 1

    return selected


if __name__ == "__main__":

    # loading the data

    dataset_name = "tanque"
    df = pd.read_csv(f"data/{dataset_name}.csv")

    if dataset_name in ["ball-and-beam", "robot-arm"]:
        u = df["u"].values
        y = df["y"].values
    elif dataset_name == "exchanger":
        u = df["q"].values
        y = df["th"].values
    elif dataset_name == "SNLS80mV":
        u = df["V1"].values
        y = df["V2"].values
    elif dataset_name == "tanque":
        u_train, y_train = df["uEst"].values, df["yEst"].values
        u_test, y_test = df["uVal"].values, df["yVal"].values

    # separate the data into training and testing sets

    if dataset_name != "tanque":

        plot_io(
            u=u, y=y, title=dataset_name.replace('-', ' ')
        )

        n_train = int(0.2 * df.shape[0])
        u_train, u_test = u[-n_train:], u[:-n_train]
        y_train, y_test = y[-n_train:], y[:-n_train]

    # creating the data matrix for the train set

    nu, ny, ne = 2, 2, 0
    dm = sd.data_matrix(u=u_train, y=y_train, nu=nu, ny=ny, ne=ne)

    # creating the candidate matrix for the train set
    l = 3
    cm, comb = sd.candidate_matrix(dm, l)

    Y = y_train[:-max(nu, ny, ne)]

    tol_dict = {
        "exchanger": 0.001,
        "ball-and-beam": 0.0001,
        "robot-arm": 0.001,
        "tanque": 0.00001,
        "SNLS80mV": 0.001
    }

    # structure selection with frols
    selected = frols(cm, Y, tol_dict[dataset_name])

    # parameter estimation on train set
    P = cm[:, selected]

    T = np.linalg.inv(P.T @ P) @ P.T @ Y

    y_pred = P @ T

    plot_y(
        y=Y, y_pred=y_pred,
        title=f"{dataset_name.replace('-', ' ')} - Train"
    )

    print("Selected terms: %d of %d" % (len(selected), len(comb)))
    for i in selected:
        print(sd.get_model_term(comb[i], nu, ny, ne))

    # parameter estimation on test set

    dm = sd.data_matrix(u=u_test, y=y_test, nu=nu, ny=ny, ne=ne)

    # creating the candidate matrix for the test set

    cm, comb = sd.candidate_matrix(dm, l)

    Y = y_test[:-max(nu, ny, ne)]

    # parameter estimation on train set
    P = cm[:, selected]

    y_pred = P @ T

    plot_y(
        y=Y, y_pred=y_pred,
        title=f"{dataset_name.replace('-', ' ')} - Validation"
    )

    # computes mean squared error

    mse = np.mean((y_pred - Y)**2)
    print("MSE=%f" % (mse))

    plt.show()
