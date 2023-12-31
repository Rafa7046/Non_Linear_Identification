import numpy as np
import sysid as sd

from utils import plot_y

class Semp:
    def __init__(self, u, y, l, nu, ny, ne):
        self.maxu = max(abs(u))
        self.maxy = max(abs(y))
        self.u = u/max(abs(u))
        self.y = y/max(abs(y))
        self.nu = nu
        self.ny = ny
        self.ne = ne
        self.l = l
        self.t = np.arange(0, len(self.u) * 0.1, 0.1)
        self.limit = -max(self.nu, self.ny, self.ne)

    def __plot(self, t, y, y_hat, title, error):
        plot_y(y=y[: self.limit]*self.maxy, y_pred=y_hat*self.maxy, title=title)
        print(f"MSE of the Model = {error}")
        print("=" * 30)

    def __msse(self, y, y_hat):
        squared_errors = np.power(y - y_hat, 2)
        mse = np.mean(squared_errors)

        return mse

    def __get_candidates(self, u_test, y_test, comb):
        data_matrix = sd.data_matrix(u_test, y_test, self.nu, self.ny, self.ne)
        cm_test = sd.candidate_matrix(data_matrix, self.l)
        candidates_test = np.array([])
        for i in range(len(cm_test[1])):
            if cm_test[1][i] in comb:
                if len(candidates_test) == 0:
                    candidates_test = cm_test[0][:, i]
                else:
                    candidates_test = np.column_stack(
                        (candidates_test, cm_test[0][:, i])
                    )

        return candidates_test

    def __prediction(self, psi_in, psi_out, y_train, i):
        tup = (np.array(psi_in.copy()), psi_out[:, i])
        aux = np.column_stack(tup) if i != 0 else psi_out[:, i].copy()
        iteractions = 1 if self.ne == 0 else self.ne - 1

        for j in range(iteractions):
            if i != 0:
                theta = np.linalg.inv(aux.T @ aux) @ aux.T @ y_train
                y_hat = aux @ theta
            else:
                theta = (1 / np.dot(aux.T, aux)) * aux.T @ y_train
                y_hat = aux * theta

            e = y_train - y_hat
            if self.ne != 0:
                psi_out[:, self.nu + self.ny + 1 + j] = e

        return y_hat, aux

    def __run_semp(self, data_matrix, y_train):
        y_train = y_train[: self.limit]
        psi_in = np.array([[]])
        comb_in = []
        psi_out, comb_out = sd.candidate_matrix(data_matrix, self.l)
        J = np.inf
        offset = 0
        i = 0

        while i < (psi_out.shape[1]):
            y_hat, aux = self.__prediction(psi_in, psi_out, y_train, i)

            Ji = self.__msse(y_train, y_hat) + 0.000001*(len(comb_in)-offset)
            if Ji < J:
                J = Ji
                psi_in = aux.copy()
                comb_in.append(comb_out[i])
                i += 1
            else:
                if i >= self.ny:
                    break
                J = np.inf
                i = self.ny
                offset = len(comb_in)

        try:
            theta = np.linalg.inv(psi_in.T @ psi_in) @ psi_in.T @ y_train
            y_hat = aux @ theta
        except:
            theta = (1 / np.dot(psi_in.T, psi_in)) * psi_in.T @ y_train
            y_hat = psi_in * theta

        J = self.__msse(y_train, y_hat)

        i = 0
        while i < (psi_in.shape[1]):
            aux = psi_in.copy()
            aux = np.delete(aux, i, 1)
            theta = np.linalg.inv(aux.T @ aux) @ aux.T @ y_train
            y_hat = aux @ theta
            Ji = self.__msse(y_train, y_hat)
            srr = (J - Ji)/np.mean(np.power(y_train, 2))
            if srr > 0:
                J = Ji
                psi_in = np.delete(psi_in, i, 1)
                comb_in.pop(i)
            else:
                i += 1

        return psi_in, comb_in

    def __run_train(self, u, y):
        data_matrix = sd.data_matrix(u, y, self.nu, self.ny, self.ne)
        psi, comb = self.__run_semp(data_matrix, y)

        theta = np.linalg.inv(psi.T @ psi) @ psi.T @ y[: self.limit]
        y_hat = psi @ theta

        return y_hat, theta, comb

    def __validate(self, title):
        n_train = int(0.2 * len(self.u))

        t_train, t_test = self.t[-n_train:], self.t[:-n_train]
        u_train, u_test = self.u[-n_train:], self.u[:-n_train]
        y_train, y_test = self.y[-n_train:], self.y[:-n_train]

        y_hat, theta, comb = self.__run_train(u_train, y_train)
        y_hat_test = self.__get_candidates(u_test, y_test, comb) @ theta

        self.__plot(
            t_train,
            y_train,
            y_hat,
            title + " - Train",
            self.__msse(y_train[: self.limit], y_hat),
        )
        self.__plot(
            t_test,
            y_test,
            y_hat_test,
            title + " - Validation",
            self.__msse(y_test[: self.limit], y_hat_test),
        )

        return y_hat, y_hat_test, theta, comb, self.__msse(y_test[: self.limit], y_hat_test)

    def run(self, validation=0, title=""):
        if validation:
            y_hat, y_hat_test, theta, comb, error = self.__validate(title)
            self.y_hat_test = y_hat_test

        else:
            y_hat, theta, comb = self.__run_train(self.u, self.y)
            self.__plot(
                self.t, self.y, y_hat, title, self.__msse(self.y[: self.limit], y_hat)
            )
            error = self.__msse(self.y[: self.limit], y_hat)

        print("=" * 30)
        print(f"Non linear Regressors:")
        for i in range(len(comb)):
            print(sd.get_model_term(comb[i], self.nu, self.ny, self.ne))
        print("=" * 30)

        self.y_hat = y_hat
        self.theta = theta
        self.regressors = comb

        return error 
