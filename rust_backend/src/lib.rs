mod linalg;
mod sysid;
mod semp;
mod frols;
mod gram_schmidt;

use pyo3::prelude::*;

use semp::run_semp;
use frols::run_frols;
use gram_schmidt::run_gram_schmidt;

// ── SEMP bindings ──

#[pyclass]
struct PySempResult {
    #[pyo3(get)]
    y_hat: Vec<f64>,
    #[pyo3(get)]
    y_hat_test: Option<Vec<f64>>,
    #[pyo3(get)]
    theta: Vec<f64>,
    #[pyo3(get)]
    regressors: Vec<Vec<usize>>,
    #[pyo3(get)]
    mse_train: f64,
    #[pyo3(get)]
    mse_test: Option<f64>,
    #[pyo3(get)]
    regressor_names: Vec<String>,
}

#[pyfunction]
fn py_run_semp(
    u: Vec<f64>,
    y: Vec<f64>,
    l: usize,
    nu: usize,
    ny: usize,
    ne: usize,
    validation: bool,
) -> PyResult<PySempResult> {
    let result = run_semp(&u, &y, l, nu, ny, ne, validation);
    Ok(PySempResult {
        y_hat: result.y_hat,
        y_hat_test: result.y_hat_test,
        theta: result.theta,
        regressors: result.regressors,
        mse_train: result.mse_train,
        mse_test: result.mse_test,
        regressor_names: result.regressor_names,
    })
}

// ── FROLS bindings ──

#[pyclass]
struct PyFrolsResult {
    #[pyo3(get)]
    y_hat_train: Vec<f64>,
    #[pyo3(get)]
    y_hat_test: Option<Vec<f64>>,
    #[pyo3(get)]
    theta: Vec<f64>,
    #[pyo3(get)]
    selected_indices: Vec<usize>,
    #[pyo3(get)]
    regressors: Vec<Vec<usize>>,
    #[pyo3(get)]
    mse_train: f64,
    #[pyo3(get)]
    mse_test: Option<f64>,
    #[pyo3(get)]
    regressor_names: Vec<String>,
    #[pyo3(get)]
    err: Vec<f64>,
}

#[pyfunction]
fn py_run_frols(
    u: Vec<f64>,
    y: Vec<f64>,
    nu: usize,
    ny: usize,
    ne: usize,
    nl: usize,
    tol: f64,
    max_iter: usize,
    validation: bool,
) -> PyResult<PyFrolsResult> {
    let result = run_frols(&u, &y, nu, ny, ne, nl, tol, max_iter, validation);
    Ok(PyFrolsResult {
        y_hat_train: result.y_hat_train,
        y_hat_test: result.y_hat_test,
        theta: result.theta,
        selected_indices: result.selected_indices,
        regressors: result.regressors,
        mse_train: result.mse_train,
        mse_test: result.mse_test,
        regressor_names: result.regressor_names,
        err: result.err,
    })
}

// ── Module ──

#[pymodule]
fn rust_nlsi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_run_semp, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_frols, m)?)?;
    m.add_function(wrap_pyfunction!(py_run_gram_schmidt, m)?)?;
    m.add_class::<PySempResult>()?;
    m.add_class::<PyFrolsResult>()?;
    m.add_class::<PyGramSchmidtResult>()?;
    Ok(())
}

// ── Gram-Schmidt bindings ──

#[pyclass]
struct PyGramSchmidtResult {
    #[pyo3(get)]
    full_y_hat_train: Vec<f64>,
    #[pyo3(get)]
    full_y_hat_test: Option<Vec<f64>>,
    #[pyo3(get)]
    full_theta: Vec<f64>,
    #[pyo3(get)]
    full_mse_train: f64,
    #[pyo3(get)]
    full_mse_test: Option<f64>,
    #[pyo3(get)]
    full_regressor_names: Vec<String>,

    #[pyo3(get)]
    sel_y_hat_train: Vec<f64>,
    #[pyo3(get)]
    sel_y_hat_test: Option<Vec<f64>>,
    #[pyo3(get)]
    sel_theta: Vec<f64>,
    #[pyo3(get)]
    sel_mse_train: f64,
    #[pyo3(get)]
    sel_mse_test: Option<f64>,
    #[pyo3(get)]
    sel_regressor_names: Vec<String>,
    #[pyo3(get)]
    selected_indices: Vec<usize>,
    #[pyo3(get)]
    err: Vec<f64>,
}

#[pyfunction]
fn py_run_gram_schmidt(
    u: Vec<f64>,
    y: Vec<f64>,
    nu: usize,
    ny: usize,
    nl: usize,
    n_theta: usize,
    validation: bool,
) -> PyResult<PyGramSchmidtResult> {
    let result = run_gram_schmidt(&u, &y, nu, ny, nl, n_theta, validation);
    Ok(PyGramSchmidtResult {
        full_y_hat_train: result.full_y_hat_train,
        full_y_hat_test: result.full_y_hat_test,
        full_theta: result.full_theta,
        full_mse_train: result.full_mse_train,
        full_mse_test: result.full_mse_test,
        full_regressor_names: result.full_regressor_names,

        sel_y_hat_train: result.sel_y_hat_train,
        sel_y_hat_test: result.sel_y_hat_test,
        sel_theta: result.sel_theta,
        sel_mse_train: result.sel_mse_train,
        sel_mse_test: result.sel_mse_test,
        sel_regressor_names: result.sel_regressor_names,
        selected_indices: result.selected_indices,
        err: result.err,
    })
}
