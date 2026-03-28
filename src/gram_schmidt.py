import numpy as np
from rust_nlsi import py_run_gram_schmidt
from src.plotting import plot_y


class GramSchmidt:
    def __init__(self, u, y, nu, ny, nl=3, n_theta=4):
        self.u = u
        self.y = y
        self.nu = nu
        self.ny = ny
        self.nl = nl
        self.n_theta = n_theta
        self.limit = -max(self.nu, self.ny)

    def _plot(self, y, y_hat, title, error):
        plot_y(y=y[: self.limit], y_pred=y_hat, title=title)
        print(f"MSE of the Model = {error}")
        print("=" * 30)

    def run(self, validation=False, title=""):
        result = py_run_gram_schmidt(
            self.u.tolist(),
            self.y.tolist(),
            self.nu,
            self.ny,
            self.nl,
            self.n_theta,
            bool(validation),
        )

        self.full_theta = np.array(result.full_theta)
        self.sel_theta = np.array(result.sel_theta)
        self.selected_indices = result.selected_indices
        self.err = result.err

        self.full_y_hat = np.array(result.full_y_hat_train)
        self.sel_y_hat = np.array(result.sel_y_hat_train)

        if validation:
            self.full_y_hat_test = np.array(result.full_y_hat_test)
            self.sel_y_hat_test = np.array(result.sel_y_hat_test)

            n_train = int(0.2 * len(self.u))
            y_train = self.y[-n_train:]
            y_test = self.y[:-n_train]

            print("── Full Model ──")
            self._plot(y_train, self.full_y_hat, title + " - Full (Train)", result.full_mse_train)
            self._plot(y_test, self.full_y_hat_test, title + " - Full (Test)", result.full_mse_test)

            print("── Selected Model ──")
            self._plot(y_train, self.sel_y_hat, title + " - Selected (Train)", result.sel_mse_train)
            self._plot(y_test, self.sel_y_hat_test, title + " - Selected (Test)", result.sel_mse_test)

            error = result.sel_mse_test
        else:
            print("── Full Model ──")
            self._plot(self.y, self.full_y_hat, title + " - Full", result.full_mse_train)

            print("── Selected Model ──")
            self._plot(self.y, self.sel_y_hat, title + " - Selected", result.sel_mse_train)

            error = result.sel_mse_train

        print(f"Full model regressors ({len(result.full_regressor_names)}):")
        for name in result.full_regressor_names:
            print(f"  {name}")
        print(f"Full theta: {self.full_theta}")
        print("=" * 30)

        print(f"Selected regressors ({len(result.sel_regressor_names)}):")
        for i, name in enumerate(result.sel_regressor_names):
            print(f"  {name}  (ERR={self.err[i]:.6f})")
        print(f"ESR: {1 - sum(self.err):.6f}")
        print(f"Selected theta: {self.sel_theta}")
        print("=" * 30)

        return error
