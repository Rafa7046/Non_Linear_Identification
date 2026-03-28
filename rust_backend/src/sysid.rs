use ndarray::{Array1, Array2, Axis, concatenate};
use rand::Rng;

/// Generate combinations with repetition of `k` elements from `arr`.
fn combine(arr: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if arr.is_empty() {
        return vec![];
    }
    let head = arr[0];
    let tail = &arr[1..];

    let without_head = combine(tail, k);
    let with_head_partial = combine(arr, k - 1);
    let with_head: Vec<Vec<usize>> = with_head_partial
        .into_iter()
        .map(|mut v| {
            v.insert(0, head);
            v
        })
        .collect();

    let mut result = with_head;
    result.extend(without_head);
    result
}

/// Generate all combinations of elements from 0..num_cols up to length `size`.
pub fn generate_combinations(num_cols: usize, size: usize) -> Vec<Vec<usize>> {
    let arr: Vec<usize> = (0..num_cols).collect();
    let mut answer = Vec::new();
    for k in 1..=size {
        for comb in combine(&arr, k) {
            answer.push(comb);
        }
    }
    answer
}

/// Build the data matrix Ψ from input u and output y.
pub fn data_matrix(
    u: &Array1<f64>,
    y: &Array1<f64>,
    nu: usize,
    ny: usize,
    ne: usize,
) -> Array2<f64> {
    let n_data = u.len();
    let n = nu.max(ny).max(ne);
    let rows = n_data - n;

    let mut u_mat = Array2::<f64>::zeros((rows, nu));
    let mut y_mat = Array2::<f64>::zeros((rows, ny));

    for i in 0..nu {
        let col_idx = nu - 1 - i;
        for r in 0..rows {
            u_mat[[r, col_idx]] = u[i + 1 + r];
        }
    }

    for i in 0..ny {
        let col_idx = ny - 1 - i;
        for r in 0..rows {
            y_mat[[r, col_idx]] = y[i + 1 + r];
        }
    }

    if ne == 0 {
        return concatenate![Axis(1), y_mat, u_mat];
    }

    let mut e_mat = Array2::<f64>::zeros((rows, ne));
    let mut rng = rand::thread_rng();
    let e_vec: Vec<f64> = (0..n_data).map(|_| rng.gen::<f64>()).collect();

    for i in 0..ne {
        let col_idx = ne - 1 - i;
        for r in 0..rows {
            e_mat[[r, col_idx]] = e_vec[i + 1 + r];
        }
    }

    concatenate![Axis(1), y_mat, u_mat, e_mat]
}

/// Build candidate matrix from data matrix with non-linearity degree `nl`.
/// Returns (candidate_matrix, combinations).
pub fn candidate_matrix(
    dm: &Array2<f64>,
    nl: usize,
) -> (Array2<f64>, Vec<Vec<usize>>) {
    let combinations = generate_combinations(dm.ncols(), nl);
    let rows = dm.nrows();

    let mut cm = Array2::<f64>::zeros((rows, combinations.len()));

    for (col_idx, comb) in combinations.iter().enumerate() {
        for r in 0..rows {
            let mut prod = 1.0;
            for &c in comb {
                prod *= dm[[r, c]];
            }
            cm[[r, col_idx]] = prod;
        }
    }

    (cm, combinations)
}

/// Get the model term string for given indices.
pub fn get_model_term(idxs: &[usize], nu: usize, ny: usize, ne: usize) -> String {
    let mut parts = Vec::new();
    for &i in idxs {
        if i + 1 > ny + nu {
            parts.push(format!("e[k-{}]", nu + ny + ne - i));
        } else if i + 1 > ny {
            parts.push(format!("u[k-{}]", nu + ny - i));
        } else {
            parts.push(format!("y[k-{}]", ny - i));
        }
    }
    parts.join(" ")
}
