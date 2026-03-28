use ndarray::{Array1, Array2, s};

/// Solve linear system A @ x = b via normal equations: x = (A^T A)^{-1} A^T b
/// For a single column vector (1D), special-cases the scalar solution.
pub fn lstsq(a: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
    let at = a.t();
    let ata = at.dot(a);
    let atb = at.dot(b);

    let inv = invert_matrix(&ata);
    inv.dot(&atb)
}

/// Scalar least squares: theta = (a^T a)^{-1} a^T b
pub fn lstsq_scalar(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let ata = a.dot(a);
    if ata.abs() < 1e-30 {
        return 0.0;
    }
    a.dot(b) / ata
}

/// Invert a square matrix using Gauss-Jordan elimination
pub fn invert_matrix(m: &Array2<f64>) -> Array2<f64> {
    let n = m.nrows();
    assert_eq!(n, m.ncols(), "Matrix must be square");

    // Augmented matrix [m | I]
    let mut aug = Array2::<f64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = m[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_row != col {
            for j in 0..(2 * n) {
                let tmp = aug[[col, j]];
                aug[[col, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = tmp;
            }
        }

        let pivot = aug[[col, col]];
        if pivot.abs() < 1e-30 {
            // Singular-ish matrix, add small regularization
            aug[[col, col]] += 1e-10;
            let pivot = aug[[col, col]];
            for j in 0..(2 * n) {
                aug[[col, j]] /= pivot;
            }
        } else {
            for j in 0..(2 * n) {
                aug[[col, j]] /= pivot;
            }
        }

        for row in 0..n {
            if row != col {
                let factor = aug[[row, col]];
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    aug.slice(s![.., n..]).to_owned()
}
