import numpy as np
from rust_nlsi import py_run_frols
from src.plotting import plot_y


class Frols:
    def __init__(self, u, y, nu, ny, ne=0, nl=1, tol=0.0, max_iter=10):
        self.maxu = max(abs(u))
        self.maxy = max(abs(y))
        self.u = u
        self.y = y
        self.nu = nu
        self.ny = ny
        self.ne = ne
        self.nl = nl
        self.tol = tol
        self.max_iter = max_iter
        self.limit = -max(self.nu, self.ny, max(self.ne, 1))

    def _plot(self, y, y_hat, title, error):
        plot_y(y=y[: self.limit] * self.maxy, y_pred=y_hat, title=title)
        print(f"MSE of the Model = {error}")
        print("=" * 30)

    def run(self, validation=False, title=""):
        result = py_run_frols(
            self.u.tolist(),
            self.y.tolist(),
            self.nu,
            self.ny,
            self.ne,
            self.nl,
            self.tol,
            self.max_iter,
            bool(validation),
        )

        self.y_hat = np.array(result.y_hat_train)
        self.theta = np.array(result.theta)
        self.regressors = result.regressors
        self.err = result.err

        if validation:
            self.y_hat_test = np.array(result.y_hat_test)

            n_train = int(0.2 * len(self.u))
            y_norm = self.y / self.maxy
            y_train = y_norm[-n_train:]
            y_test = y_norm[:-n_train]

            self._plot(y_train, self.y_hat, title + " - Train", result.mse_train)
            self._plot(y_test, self.y_hat_test, title + " - Validation", result.mse_test)
            error = result.mse_test
        else:
            y_norm = self.y / self.maxy
            self._plot(y_norm, self.y_hat, title, result.mse_train)
            error = result.mse_train

        print("Selected Regressors:")
        for name in result.regressor_names:
            print(f"  {name}")
        print(f"ERR: {self.err}")
        print(f"ESR: {1 - sum(self.err):.6f}")
        print(f"Theta: {self.theta}")
        print("=" * 30)

        return error
