use ndarray::{Array1, Array2};

use crate::linalg::lstsq;
use crate::sysid::{data_matrix, candidate_matrix, get_model_term};

/// Mean Squared Error
fn msse(y: &Array1<f64>, y_hat: &Array1<f64>) -> f64 {
    let diff = y - y_hat;
    let sq = &diff * &diff;
    sq.mean().unwrap_or(0.0)
}

/// FROLS result
pub struct FrolsResult {
    pub y_hat_train: Vec<f64>,
    pub y_hat_test: Option<Vec<f64>>,
    pub theta: Vec<f64>,
    pub selected_indices: Vec<usize>,
    pub regressors: Vec<Vec<usize>>,
    pub mse_train: f64,
    pub mse_test: Option<f64>,
    pub regressor_names: Vec<String>,
    pub err: Vec<f64>,
}

/// Core FROLS with Gram-Schmidt orthogonalization and ERR-based selection
fn frols_core(
    cm: &Array2<f64>,
    y: &Array1<f64>,
    tol: f64,
    max_iter: usize,
) -> (Vec<usize>, Vec<f64>) {
    let m = cm.ncols();

    // Initial candidates as column vectors
    let w1i: Vec<Array1<f64>> = (0..m).map(|i| cm.column(i).to_owned()).collect();

    let yty = y.dot(y);

    // Step 1: compute ERR for all candidates
    let g1i: Vec<f64> = w1i
        .iter()
        .map(|w| {
            let ww = w.dot(w);
            if ww.abs() < 1e-30 { 0.0 } else { w.dot(y) / ww }
        })
        .collect();

    let erri: Vec<f64> = (0..m)
        .map(|i| {
            let ww = w1i[i].dot(&w1i[i]);
            g1i[i] * g1i[i] * ww / yty
        })
        .collect();

    // Select first term
    let first = erri
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let mut selected = vec![first];
    let mut selected_erri = vec![erri[first]];
    let mut w_selected: Vec<Array1<f64>> = vec![w1i[first].clone()];

    // Iterative selection
    let mut k = 1usize;
    while 1.0 - selected_erri.iter().sum::<f64>() > tol && k < max_iter {
        let mut err_k = vec![0.0f64; m];

        for i in 0..m {
            if selected.contains(&i) {
                continue;
            }

            // Gram-Schmidt: orthogonalize w1i[i] against W
            let mut wki = w1i[i].clone();
            for j in 0..k {
                let wjwj = w_selected[j].dot(&w_selected[j]);
                if wjwj.abs() < 1e-30 {
                    continue;
                }
                let alpha = w_selected[j].dot(&w1i[i]) / wjwj;
                wki = &wki - &(&w_selected[j] * alpha);
            }

            let wki_wki = wki.dot(&wki);
            if wki_wki.abs() < 1e-30 {
                continue;
            }
            let gki = wki.dot(y) / wki_wki;
            err_k[i] = gki * gki * wki_wki / yty;
        }

        let best = err_k
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        selected.push(best);

        // Compute orthogonalized W for new selection
        let mut w_new = w1i[best].clone();
        for j in 0..k {
            let wjwj = w_selected[j].dot(&w_selected[j]);
            if wjwj.abs() < 1e-30 {
                continue;
            }
            let alpha = w_selected[j].dot(&w1i[best]) / wjwj;
            w_new = &w_new - &(&w_selected[j] * alpha);
        }
        w_selected.push(w_new);
        selected_erri.push(err_k[best]);
        k += 1;
    }

    // Remove last term (following original Python logic)
    selected.pop();
    selected_erri.pop();

    (selected, selected_erri)
}

/// Run FROLS algorithm
pub fn run_frols(
    u: &[f64],
    y: &[f64],
    nu: usize,
    ny: usize,
    ne: usize,
    nl: usize,
    tol: f64,
    max_iter: usize,
    validation: bool,
) -> FrolsResult {
    let u_arr = Array1::from_vec(u.to_vec());
    let y_arr = Array1::from_vec(y.to_vec());

    let maxu = u_arr.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max);
    let maxy = y_arr.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max);

    let u_norm = &u_arr / maxu;
    let y_norm = &y_arr / maxy;

    let limit_val = nu.max(ny).max(if ne == 0 { 1 } else { ne });

    if validation {
        run_frols_validation(&u_norm, &y_norm, nu, ny, ne, nl, tol, max_iter, limit_val, maxy)
    } else {
        run_frols_no_validation(&u_norm, &y_norm, nu, ny, ne, nl, tol, max_iter, limit_val, maxy)
    }
}

fn run_frols_no_validation(
    u: &Array1<f64>,
    y: &Array1<f64>,
    nu: usize,
    ny: usize,
    ne: usize,
    nl: usize,
    tol: f64,
    max_iter: usize,
    limit_val: usize,
    maxy: f64,
) -> FrolsResult {
    let dm = data_matrix(u, y, nu, ny, ne);
    let (cm, comb) = candidate_matrix(&dm, nl);

    let limit = y.len() - limit_val;
    let y_target = Array1::from_vec(y.iter().take(limit).cloned().collect());

    let (selected, err) = frols_core(&cm, &y_target, tol, max_iter);

    // Build selected candidate matrix
    let rows = cm.nrows();
    let mut p = Array2::<f64>::zeros((rows, selected.len()));
    for (j, &col_idx) in selected.iter().enumerate() {
        p.column_mut(j).assign(&cm.column(col_idx));
    }

    let theta_arr = lstsq(&p, &y_target);
    let y_hat = p.dot(&theta_arr);
    let mse = msse(&y_target, &y_hat);

    let selected_combs: Vec<Vec<usize>> = selected.iter().map(|&i| comb[i].clone()).collect();
    let regressor_names: Vec<String> = selected_combs
        .iter()
        .map(|c| get_model_term(c, nu, ny, ne))
        .collect();

    FrolsResult {
        y_hat_train: y_hat.iter().map(|v| v * maxy).collect(),
        y_hat_test: None,
        theta: theta_arr.to_vec(),
        selected_indices: selected,
        regressors: selected_combs,
        mse_train: mse,
        mse_test: None,
        regressor_names,
        err,
    }
}

fn run_frols_validation(
    u: &Array1<f64>,
    y: &Array1<f64>,
    nu: usize,
    ny: usize,
    ne: usize,
    nl: usize,
    tol: f64,
    max_iter: usize,
    limit_val: usize,
    maxy: f64,
) -> FrolsResult {
    let n_train = (0.2 * u.len() as f64) as usize;
    let n_test = u.len() - n_train;

    let u_train = Array1::from_vec(u.iter().skip(n_test).cloned().collect());
    let y_train_full = Array1::from_vec(y.iter().skip(n_test).cloned().collect());
    let u_test = Array1::from_vec(u.iter().take(n_test).cloned().collect());
    let y_test_full = Array1::from_vec(y.iter().take(n_test).cloned().collect());

    let limit = y_train_full.len() - limit_val;

    // Train
    let dm_train = data_matrix(&u_train, &y_train_full, nu, ny, ne);
    let (cm_train, comb_train) = candidate_matrix(&dm_train, nl);
    let y_target = Array1::from_vec(y_train_full.iter().take(limit).cloned().collect());

    let (selected, err) = frols_core(&cm_train, &y_target, tol, max_iter);

    // Parameter estimation on train
    let rows_train = cm_train.nrows();
    let mut p_train = Array2::<f64>::zeros((rows_train, selected.len()));
    for (j, &col_idx) in selected.iter().enumerate() {
        p_train.column_mut(j).assign(&cm_train.column(col_idx));
    }

    let theta_arr = lstsq(&p_train, &y_target);
    let y_hat_train = p_train.dot(&theta_arr);
    let mse_train = msse(&y_target, &y_hat_train);

    // Test
    let dm_test = data_matrix(&u_test, &y_test_full, nu, ny, ne);
    let (cm_test, comb_test) = candidate_matrix(&dm_test, nl);
    let limit_test = y_test_full.len() - limit_val;
    let y_test_target = Array1::from_vec(y_test_full.iter().take(limit_test).cloned().collect());

    // Map selected columns from train to test via combination matching
    let selected_combs: Vec<Vec<usize>> = selected.iter().map(|&i| comb_train[i].clone()).collect();
    let mut test_col_indices: Vec<usize> = Vec::new();
    for sc in &selected_combs {
        for (idx, ct) in comb_test.iter().enumerate() {
            if ct == sc {
                test_col_indices.push(idx);
                break;
            }
        }
    }

    let rows_test = cm_test.nrows();
    let mut p_test = Array2::<f64>::zeros((rows_test, test_col_indices.len()));
    for (j, &col_idx) in test_col_indices.iter().enumerate() {
        p_test.column_mut(j).assign(&cm_test.column(col_idx));
    }

    let y_hat_test = p_test.dot(&theta_arr);
    let mse_test = msse(&y_test_target, &y_hat_test);

    let regressor_names: Vec<String> = selected_combs
        .iter()
        .map(|c| get_model_term(c, nu, ny, ne))
        .collect();

    FrolsResult {
        y_hat_train: y_hat_train.iter().map(|v| v * maxy).collect(),
        y_hat_test: Some(y_hat_test.iter().map(|v| v * maxy).collect()),
        theta: theta_arr.to_vec(),
        selected_indices: selected,
        regressors: selected_combs,
        mse_train,
        mse_test: Some(mse_test),
        regressor_names,
        err,
    }
}
