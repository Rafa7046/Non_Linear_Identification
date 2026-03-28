use ndarray::{Array1, Array2};

use crate::linalg::lstsq;
use crate::sysid::{data_matrix, candidate_matrix, get_model_term};

fn msse(y: &Array1<f64>, y_hat: &Array1<f64>) -> f64 {
    let diff = y - y_hat;
    let sq = &diff * &diff;
    sq.mean().unwrap_or(0.0)
}

pub struct GramSchmidtResult {
    // Full model
    pub full_y_hat_train: Vec<f64>,
    pub full_y_hat_test: Option<Vec<f64>>,
    pub full_theta: Vec<f64>,
    pub full_mse_train: f64,
    pub full_mse_test: Option<f64>,
    pub full_regressor_names: Vec<String>,

    // Selected model
    pub sel_y_hat_train: Vec<f64>,
    pub sel_y_hat_test: Option<Vec<f64>>,
    pub sel_theta: Vec<f64>,
    pub sel_mse_train: f64,
    pub sel_mse_test: Option<f64>,
    pub sel_regressor_names: Vec<String>,
    pub selected_indices: Vec<usize>,
    pub err: Vec<f64>,
}

/// Core Gram-Schmidt structure selection: select exactly `n_theta` terms
fn gs_select(
    cm: &Array2<f64>,
    y: &Array1<f64>,
    n_theta: usize,
) -> (Vec<usize>, Vec<f64>) {
    let m = cm.ncols();

    let w1i: Vec<Array1<f64>> = (0..m).map(|i| cm.column(i).to_owned()).collect();
    let yty = y.dot(y);

    // Step 1: ERR for all candidates
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
    let mut selected_err = vec![erri[first]];
    let mut w_selected: Vec<Array1<f64>> = vec![w1i[first].clone()];

    // Iterative selection
    for k in 1..n_theta {
        let mut err_k = vec![0.0f64; m];

        for i in 0..m {
            if selected.contains(&i) {
                continue;
            }

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
        selected_err.push(err_k[best]);
    }

    (selected, selected_err)
}

/// Extract selected columns into a new matrix
fn select_columns(cm: &Array2<f64>, selected: &[usize]) -> Array2<f64> {
    let rows = cm.nrows();
    let mut p = Array2::<f64>::zeros((rows, selected.len()));
    for (j, &col_idx) in selected.iter().enumerate() {
        p.column_mut(j).assign(&cm.column(col_idx));
    }
    p
}

pub fn run_gram_schmidt(
    u: &[f64],
    y: &[f64],
    nu: usize,
    ny: usize,
    nl: usize,
    n_theta: usize,
    validation: bool,
) -> GramSchmidtResult {
    let u_arr = Array1::from_vec(u.to_vec());
    let y_arr = Array1::from_vec(y.to_vec());

    let limit_val = nu.max(ny);

    if validation {
        run_gs_validation(&u_arr, &y_arr, nu, ny, nl, n_theta, limit_val)
    } else {
        run_gs_no_validation(&u_arr, &y_arr, nu, ny, nl, n_theta, limit_val)
    }
}

fn run_gs_no_validation(
    u: &Array1<f64>,
    y: &Array1<f64>,
    nu: usize,
    ny: usize,
    nl: usize,
    n_theta: usize,
    limit_val: usize,
) -> GramSchmidtResult {
    let dm = data_matrix(u, y, nu, ny, 0);
    let (cm, comb) = candidate_matrix(&dm, nl);

    let limit = y.len() - limit_val;
    let y_target = Array1::from_vec(y.iter().take(limit).cloned().collect());

    // Full model
    let full_theta = lstsq(&cm, &y_target);
    let full_y_hat = cm.dot(&full_theta);
    let full_mse = msse(&y_target, &full_y_hat);

    let full_regressor_names: Vec<String> = comb
        .iter()
        .map(|c| get_model_term(c, nu, ny, 0))
        .collect();

    // Gram-Schmidt selection
    let (selected, err) = gs_select(&cm, &y_target, n_theta);

    let p = select_columns(&cm, &selected);
    let sel_theta = lstsq(&p, &y_target);
    let sel_y_hat = p.dot(&sel_theta);
    let sel_mse = msse(&y_target, &sel_y_hat);

    let sel_combs: Vec<Vec<usize>> = selected.iter().map(|&i| comb[i].clone()).collect();
    let sel_regressor_names: Vec<String> = sel_combs
        .iter()
        .map(|c| get_model_term(c, nu, ny, 0))
        .collect();

    GramSchmidtResult {
        full_y_hat_train: full_y_hat.to_vec(),
        full_y_hat_test: None,
        full_theta: full_theta.to_vec(),
        full_mse_train: full_mse,
        full_mse_test: None,
        full_regressor_names,

        sel_y_hat_train: sel_y_hat.to_vec(),
        sel_y_hat_test: None,
        sel_theta: sel_theta.to_vec(),
        sel_mse_train: sel_mse,
        sel_mse_test: None,
        sel_regressor_names,
        selected_indices: selected,
        err,
    }
}

fn run_gs_validation(
    u: &Array1<f64>,
    y: &Array1<f64>,
    nu: usize,
    ny: usize,
    nl: usize,
    n_theta: usize,
    limit_val: usize,
) -> GramSchmidtResult {
    let n_train = (0.2 * u.len() as f64) as usize;
    let n_test = u.len() - n_train;

    let u_train = Array1::from_vec(u.iter().skip(n_test).cloned().collect());
    let y_train_full = Array1::from_vec(y.iter().skip(n_test).cloned().collect());
    let u_test = Array1::from_vec(u.iter().take(n_test).cloned().collect());
    let y_test_full = Array1::from_vec(y.iter().take(n_test).cloned().collect());

    let limit_train = y_train_full.len() - limit_val;
    let limit_test = y_test_full.len() - limit_val;

    // Train matrices
    let dm_train = data_matrix(&u_train, &y_train_full, nu, ny, 0);
    let (cm_train, comb_train) = candidate_matrix(&dm_train, nl);
    let y_train_target = Array1::from_vec(y_train_full.iter().take(limit_train).cloned().collect());

    // Test matrices
    let dm_test = data_matrix(&u_test, &y_test_full, nu, ny, 0);
    let (cm_test, comb_test) = candidate_matrix(&dm_test, nl);
    let y_test_target = Array1::from_vec(y_test_full.iter().take(limit_test).cloned().collect());

    // ── Full model ──
    let full_theta = lstsq(&cm_train, &y_train_target);
    let full_y_hat_train = cm_train.dot(&full_theta);
    let full_mse_train = msse(&y_train_target, &full_y_hat_train);

    let full_y_hat_test = cm_test.dot(&full_theta);
    let full_mse_test = msse(&y_test_target, &full_y_hat_test);

    let full_regressor_names: Vec<String> = comb_train
        .iter()
        .map(|c| get_model_term(c, nu, ny, 0))
        .collect();

    // ── Gram-Schmidt selection ──
    let (selected, err) = gs_select(&cm_train, &y_train_target, n_theta);

    // Selected model on train
    let p_train = select_columns(&cm_train, &selected);
    let sel_theta = lstsq(&p_train, &y_train_target);
    let sel_y_hat_train = p_train.dot(&sel_theta);
    let sel_mse_train = msse(&y_train_target, &sel_y_hat_train);

    // Map selected columns to test via combination matching
    let sel_combs: Vec<Vec<usize>> = selected.iter().map(|&i| comb_train[i].clone()).collect();
    let mut test_col_indices: Vec<usize> = Vec::new();
    for sc in &sel_combs {
        for (idx, ct) in comb_test.iter().enumerate() {
            if ct == sc {
                test_col_indices.push(idx);
                break;
            }
        }
    }

    let p_test = select_columns(&cm_test, &test_col_indices);
    let sel_y_hat_test = p_test.dot(&sel_theta);
    let sel_mse_test = msse(&y_test_target, &sel_y_hat_test);

    let sel_regressor_names: Vec<String> = sel_combs
        .iter()
        .map(|c| get_model_term(c, nu, ny, 0))
        .collect();

    GramSchmidtResult {
        full_y_hat_train: full_y_hat_train.to_vec(),
        full_y_hat_test: Some(full_y_hat_test.to_vec()),
        full_theta: full_theta.to_vec(),
        full_mse_train,
        full_mse_test: Some(full_mse_test),
        full_regressor_names,

        sel_y_hat_train: sel_y_hat_train.to_vec(),
        sel_y_hat_test: Some(sel_y_hat_test.to_vec()),
        sel_theta: sel_theta.to_vec(),
        sel_mse_train,
        sel_mse_test: Some(sel_mse_test),
        sel_regressor_names,
        selected_indices: selected,
        err,
    }
}
