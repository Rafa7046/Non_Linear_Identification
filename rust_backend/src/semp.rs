use ndarray::{Array1, Array2, Axis, s, concatenate};

use crate::linalg::{lstsq, lstsq_scalar};
use crate::sysid::{data_matrix, candidate_matrix, get_model_term};

/// Mean Squared Error
fn msse(y: &Array1<f64>, y_hat: &Array1<f64>) -> f64 {
    let diff = y - y_hat;
    let sq = &diff * &diff;
    sq.mean().unwrap_or(0.0)
}

/// Prediction step: builds aux matrix from psi_in + column i of psi_out, solves for theta
fn prediction(
    psi_in: &Option<Array2<f64>>,
    psi_out: &mut Array2<f64>,
    y_train: &Array1<f64>,
    i: usize,
    ne: usize,
    nu: usize,
    ny: usize,
) -> (Array1<f64>, Array2<f64>) {
    let col_i = psi_out.column(i).to_owned();

    let aux: Array2<f64> = match psi_in {
        Some(ref mat) if mat.ncols() > 0 => {
            // Concatenate psi_in columns with column i
            let col_2d = col_i.insert_axis(Axis(1));
            concatenate![Axis(1), *mat, col_2d]
        }
        _ => {
            // Just column i as a 2D matrix
            col_i.clone().insert_axis(Axis(1))
        }
    };

    let iterations = if ne == 0 { 1 } else { ne };
    let aux_mut = aux.clone();
    let mut y_hat = Array1::<f64>::zeros(y_train.len());

    for j in 0..iterations {
        if aux_mut.ncols() > 1 || psi_in.is_some() && psi_in.as_ref().unwrap().ncols() > 0 {
            let theta = lstsq(&aux_mut, y_train);
            y_hat = aux_mut.dot(&theta);
        } else {
            let col = aux_mut.column(0).to_owned();
            let theta_scalar = lstsq_scalar(&col, y_train);
            y_hat = &col * theta_scalar;
        }

        if ne != 0 {
            let e = y_train - &y_hat;
            let e_col_idx = nu + ny + 1 + j;
            if e_col_idx < psi_out.ncols() {
                psi_out.column_mut(e_col_idx).assign(&e);
            }
        }
    }

    (y_hat, aux_mut)
}

/// Result of running the SEMP algorithm
pub struct SempResult {
    pub y_hat: Vec<f64>,
    pub y_hat_test: Option<Vec<f64>>,
    pub theta: Vec<f64>,
    pub regressors: Vec<Vec<usize>>,
    pub mse_train: f64,
    pub mse_test: Option<f64>,
    pub regressor_names: Vec<String>,
}

/// Run the full SEMP algorithm
pub fn run_semp(
    u: &[f64],
    y: &[f64],
    l: usize,
    nu: usize,
    ny: usize,
    ne: usize,
    validation: bool,
) -> SempResult {
    let u_arr = Array1::from_vec(u.to_vec());
    let y_arr = Array1::from_vec(y.to_vec());

    let maxu = u_arr.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max);
    let maxy = y_arr.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max);

    let u_norm = &u_arr / maxu;
    let y_norm = &y_arr / maxy;

    let limit_val = nu.max(ny).max(ne);

    if validation {
        run_with_validation(&u_norm, &y_norm, l, nu, ny, ne, limit_val, maxy)
    } else {
        run_without_validation(&u_norm, &y_norm, l, nu, ny, ne, limit_val, maxy)
    }
}

fn run_without_validation(
    u: &Array1<f64>,
    y: &Array1<f64>,
    l: usize,
    nu: usize,
    ny: usize,
    ne: usize,
    limit_val: usize,
    maxy: f64,
) -> SempResult {
    let dm = data_matrix(u, y, nu, ny, ne);
    let limit = y.len() - limit_val;
    let y_train = y.slice(s![..limit]).to_owned();

    let (psi, comb) = run_semp_core(&dm, &y_train, l, nu, ny, ne);

    let theta_arr = lstsq(&psi, &y_train);
    let y_hat = psi.dot(&theta_arr);

    let mse = msse(&y_train, &y_hat);

    let y_hat_scaled: Vec<f64> = y_hat.iter().map(|v| v * maxy).collect();
    let theta_vec: Vec<f64> = theta_arr.to_vec();

    let regressor_names: Vec<String> = comb
        .iter()
        .map(|c| get_model_term(c, nu, ny, ne))
        .collect();

    SempResult {
        y_hat: y_hat_scaled,
        y_hat_test: None,
        theta: theta_vec,
        regressors: comb,
        mse_train: mse,
        mse_test: None,
        regressor_names,
    }
}

fn run_with_validation(
    u: &Array1<f64>,
    y: &Array1<f64>,
    l: usize,
    nu: usize,
    ny: usize,
    ne: usize,
    limit_val: usize,
    maxy: f64,
) -> SempResult {
    let n_train = (0.2 * u.len() as f64) as usize;
    let n_test = u.len() - n_train;

    let u_train = u.slice(s![n_test..]).to_owned();
    let y_train_full = y.slice(s![n_test..]).to_owned();
    let u_test = u.slice(s![..n_test]).to_owned();
    let y_test_full = y.slice(s![..n_test]).to_owned();

    let limit = y_train_full.len() - limit_val;

    // Train
    let dm_train = data_matrix(&u_train, &y_train_full, nu, ny, ne);
    let y_train_sliced = y_train_full.slice(s![..limit]).to_owned();

    let (psi, comb) = run_semp_core(&dm_train, &y_train_sliced, l, nu, ny, ne);

    let theta_arr = lstsq(&psi, &y_train_sliced);
    let y_hat_train = psi.dot(&theta_arr);
    let mse_train = msse(&y_train_sliced, &y_hat_train);

    // Validate
    let dm_test = data_matrix(&u_test, &y_test_full, nu, ny, ne);
    let (cm_test, comb_test) = candidate_matrix(&dm_test, l);

    let limit_test = y_test_full.len() - limit_val;

    // Select columns matching the training regressors
    let mut selected_cols: Vec<usize> = Vec::new();
    for c in &comb {
        for (idx, ct) in comb_test.iter().enumerate() {
            if ct == c {
                selected_cols.push(idx);
                break;
            }
        }
    }

    let mut candidates_test = Array2::<f64>::zeros((cm_test.nrows(), selected_cols.len()));
    for (j, &col_idx) in selected_cols.iter().enumerate() {
        candidates_test.column_mut(j).assign(&cm_test.column(col_idx));
    }

    let y_hat_test = candidates_test.dot(&theta_arr);
    let y_test_sliced = y_test_full.slice(s![..limit_test]).to_owned();
    let mse_test = msse(&y_test_sliced, &y_hat_test);

    let y_hat_train_scaled: Vec<f64> = y_hat_train.iter().map(|v| v * maxy).collect();
    let y_hat_test_scaled: Vec<f64> = y_hat_test.iter().map(|v| v * maxy).collect();
    let theta_vec: Vec<f64> = theta_arr.to_vec();

    let regressor_names: Vec<String> = comb
        .iter()
        .map(|c| get_model_term(c, nu, ny, ne))
        .collect();

    SempResult {
        y_hat: y_hat_train_scaled,
        y_hat_test: Some(y_hat_test_scaled),
        theta: theta_vec,
        regressors: comb,
        mse_train,
        mse_test: Some(mse_test),
        regressor_names,
    }
}

/// Core SEMP algorithm: forward selection + backward elimination
fn run_semp_core(
    dm: &Array2<f64>,
    y_train: &Array1<f64>,
    l: usize,
    nu: usize,
    ny: usize,
    ne: usize,
) -> (Array2<f64>, Vec<Vec<usize>>) {
    let (mut psi_out, comb_out) = candidate_matrix(dm, l);
    let mut psi_in: Option<Array2<f64>> = None;
    let mut comb_in: Vec<Vec<usize>> = Vec::new();
    let mut j_best = f64::INFINITY;
    let mut offset: usize = 0;
    let mut i: usize = 0;

    // Forward selection
    while i < psi_out.ncols() {
        let (y_hat, aux) = prediction(&psi_in, &mut psi_out, y_train, i, ne, nu, ny);

        let ji = msse(y_train, &y_hat) + 0.000001 * (comb_in.len() as f64 - offset as f64);

        if ji < j_best {
            j_best = ji;
            psi_in = Some(aux);
            comb_in.push(comb_out[i].clone());
            i += 1;
        } else {
            if i >= ny {
                break;
            }
            j_best = f64::INFINITY;
            i = ny;
            offset = comb_in.len();
        }
    }

    // Final theta computation
    let psi = psi_in.unwrap();
    let theta_arr = if psi.ncols() > 1 {
        lstsq(&psi, y_train)
    } else {
        let col = psi.column(0).to_owned();
        let s = lstsq_scalar(&col, y_train);
        Array1::from_vec(vec![s])
    };
    let y_hat = psi.dot(&theta_arr);
    let mut j_best = msse(y_train, &y_hat);

    // Backward elimination
    let mut psi_mut = psi.clone();
    let mut i = 0;
    while i < psi_mut.ncols() {
        // Remove column i
        let n_cols = psi_mut.ncols();
        let mut aux_cols: Vec<Array1<f64>> = Vec::new();
        for c in 0..n_cols {
            if c != i {
                aux_cols.push(psi_mut.column(c).to_owned());
            }
        }

        if aux_cols.is_empty() {
            i += 1;
            continue;
        }

        let aux = ndarray::stack(
            Axis(1),
            &aux_cols.iter().map(|c| c.view()).collect::<Vec<_>>(),
        )
        .unwrap();

        let theta_aux = lstsq(&aux, y_train);
        let y_hat_aux = aux.dot(&theta_aux);
        let ji = msse(y_train, &y_hat_aux);
        let y_mean_sq = y_train.iter().map(|v| v * v).sum::<f64>() / y_train.len() as f64;
        let srr = (j_best - ji) / y_mean_sq;

        if srr > 0.0 {
            j_best = ji;
            psi_mut = aux;
            comb_in.remove(i);
        } else {
            i += 1;
        }
    }

    (psi_mut, comb_in)
}
