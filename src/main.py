import numpy as np
import pandas as pd
import sysid as sd
import matplotlib.pyplot as plt

from utils import plot_io, plot_y


def frols(cm, y, tol, max_iter):
    w1i = [cm[:, i] for i in range(cm.shape[1])]
    M = len(w1i)
    g1i = [w1i[i] @ y / (w1i[i] @ w1i[i]) for i in range(len(w1i))]
    ERRi = [g1i[i]**2 * w1i[i] @ w1i[i] / (y @ y) for i in range(len(w1i))]
    selected = [np.argmax(ERRi)]
    W = [w1i[selected[0]]]

    selected_erri = [ERRi[selected[0]]]

    k = 1
    while 1 - np.sum(selected_erri) > tol and k < max_iter:
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
        W.append(w1i[selected[k]] - sum([alpha[j] * W[j] for j in range(k)]))
        selected_erri.append(ERRi[selected[k]])
        k += 1

    selected.pop()
    selected_erri.pop()

    return selected, selected_erri


if __name__ == "__main__":

    # loading the data

    dataset_name = "ball-and-beam"
    
    if dataset_name != "uniform":
        df = pd.read_csv(f"src/data/{dataset_name}.csv")

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
    elif dataset_name == "uniform":
        u = np.random.uniform(-1, 1, 200)
        y = np.zeros_like(u)

        for k in range(2, len(y)):
            y[k] = -0.605 * y[k - 1] - 0.163 * y[k - 2]**2 + 0.588 * u[k - 1] - 0.240 * u[k - 2]

    # separate the data into training and testing sets

    if dataset_name != "tanque":
        mu, my = np.max(np.abs(u)), np.max(np.abs(y))

        u = u / mu
        y = y / my

        n_train = int(0.2 * u.shape[0])
        u_train, u_test = u[-n_train:], u[:-n_train]
        y_train, y_test = y[-n_train:], y[:-n_train]

        plot_io(
            u=u, y=y, title=dataset_name.replace('-', ' ')
        )
    else:
        mu_train, mu_test = np.max(np.abs(u_train)), np.max(np.abs(u_test))
        my_train, my_test = np.max(np.abs(y_train)), np.max(np.abs(y_test))

        u_train, u_test = u_train / mu_train, u_test / mu_test
        y_train, y_test = y_train / my_train, y_test / my_test

        plot_io(
            u=u_train, y=y_train, title=f"{dataset_name.replace('-', ' ')}"
        )

    if dataset_name == "SNLS80mV":
        dataset_name = "silverbox"

    plt.savefig(f"src/results/{dataset_name}/io.png")
    plt.show()

    # creating the data matrix for the train set

    nu, ny, ne = 5, 5, 0
    dm = sd.data_matrix(u=u_train, y=y_train, nu=nu, ny=ny, ne=ne)

    # creating the candidate matrix for the train set
    nlin = 1
    cm, comb = sd.candidate_matrix(dm, nlin)

    Y = y_train[:-max(nu, ny, ne)]

    tol = 0
    # structure selection with frols
    selected, ERR = frols(cm, Y, tol, 10)

    # parameter estimation on train set
    P = cm[:, selected]

    T = np.linalg.inv(P.T @ P) @ P.T @ Y

    y_pred = P @ T

    plot_y(
        y=Y, y_pred=y_pred,
        title=f"{dataset_name.replace('-', ' ')} - Train"
    )

    plt.savefig(f"src/results/{dataset_name}/FROLS/train.png")
    plt.show()

    print("ERRi = ", ERR)
    print("ESR = %f" % (1 - np.sum(ERR)))
    print("Selected terms: %d of %d" % (len(selected), len(comb)), "with tol = %f" % (tol))
    for i, t in zip(selected, T):
        print("\n", sd.get_model_term(comb[i], nu, ny, ne), t)

    # parameter estimation on test set

    dm = sd.data_matrix(u=u_test, y=y_test, nu=nu, ny=ny, ne=ne)

    # creating the candidate matrix for the test set

    cm, comb = sd.candidate_matrix(dm, nlin)

    Y = y_test[:-max(nu, ny, ne)]

    # parameter estimation on train set
    P = cm[:, selected]

    y_pred = P @ T

    plot_y(
        y=Y, y_pred=y_pred,
        title=f"{dataset_name.replace('-', ' ')} - Validation"
    )

    plt.savefig(f"src/results/{dataset_name}/FROLS/test.png")

    # computes mean squared error

    mse = np.mean((y_pred - Y)**2)
    print("MSE=%f" % (mse))

    plt.show()
