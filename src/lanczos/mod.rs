//! Hermitian Lanczos estimates for expensive matrix-free operators.
//!
//! The implementation deliberately uses full, two-pass reorthogonalisation.
//! This costs extra vector operations, but avoids loss of orthogonality while
//! keeping the number of expensive operator applications strictly bounded.

use bempp_rsrs::rsrs::sketch::{SampleType, SamplingSpace};
use num::NumCast;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Standard, StandardNormal};
use rlst::{dense::tools::RandScalar, prelude::*};

type Real<T> = <T as RlstScalar>::Real;

#[derive(Debug, Clone, Copy)]
pub struct LanczosOptions {
    /// Maximum number of operator applications.
    pub max_iterations: usize,
    /// Do not accept convergence before this many applications.
    pub min_iterations: usize,
    /// Relative Ritz residual tolerance.
    pub tolerance: f64,
    /// Seed for the initial Gaussian vector.
    pub seed: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct LanczosEstimate<T> {
    pub largest_eigenvalue: T,
    pub iterations: usize,
    pub residual_bound: f64,
    pub converged: bool,
}

/// Estimate the largest eigenvalue of a Hermitian positive-semidefinite operator.
///
/// `operator` is expected to be Hermitian. For spectral norms, pass the normal
/// operator `A^* A`; the square root of the returned value estimates `||A||_2`.
pub fn estimate_largest_eigenvalue_hermitian<
    Item: RlstScalar + RandScalar,
    Space: SamplingSpace<F = Item>,
    OpImpl: AsApply<Domain = Space, Range = Space>,
>(
    operator: &OpImpl,
    options: LanczosOptions,
) -> LanczosEstimate<Real<Item>>
where
    StandardNormal: Distribution<Real<Item>>,
    Standard: Distribution<Real<Item>>,
    Real<Item>: RandScalar,
    <<Space as LinearSpace>::E as ElementImpl>::Space: InnerProductSpace,
{
    assert!(options.max_iterations > 0);
    assert!(options.tolerance >= 0.0);

    let mut rng = ChaCha8Rng::seed_from_u64(options.seed);
    let mut q = SamplingSpace::zero(operator.domain());
    operator
        .domain()
        .sampling(&mut q, &mut rng, SampleType::StandardNormal);
    let q_norm = as_f64(q.norm());
    if q_norm == 0.0 {
        return zero_estimate::<Item>();
    }
    q.scale_inplace(Item::from_real(from_f64::<Item>(1.0 / q_norm)));

    let mut basis = Vec::with_capacity(options.max_iterations);
    let mut diagonal = Vec::with_capacity(options.max_iterations);
    let mut off_diagonal = Vec::with_capacity(options.max_iterations.saturating_sub(1));
    let mut previous_beta = 0.0;
    let mut best_value = 0.0;
    let mut best_residual = f64::INFINITY;

    for iteration in 0..options.max_iterations {
        basis.push(q.duplicate());
        let mut w = operator.apply(q.r(), TransMode::NoTrans);
        let alpha_scalar = q.inner_product(w.r());
        let alpha = as_f64(alpha_scalar.re());

        w.axpy_inplace(-Item::from_real(from_f64::<Item>(alpha)), q.r());
        if basis.len() >= 2 {
            w.axpy_inplace(
                -Item::from_real(from_f64::<Item>(previous_beta)),
                basis[basis.len() - 2].r(),
            );
        }

        // Two passes are intentional: classical loss of orthogonality can make
        // a largest Ritz value look converged too early.
        for _ in 0..2 {
            for vector in &basis {
                let coefficient = vector.inner_product(w.r());
                w.axpy_inplace(-coefficient, vector.r());
            }
        }

        diagonal.push(alpha);
        let beta = as_f64(w.norm());
        let (ritz_value, last_component) = largest_ritz_pair(&diagonal, &off_diagonal);
        let residual = beta * last_component.abs();
        best_value = ritz_value.max(0.0);
        best_residual = residual;

        let completed = iteration + 1;
        let scale = best_value.abs().max(1.0);
        if completed >= options.min_iterations.min(options.max_iterations)
            && residual <= options.tolerance * scale
        {
            return LanczosEstimate {
                largest_eigenvalue: from_f64::<Item>(best_value),
                iterations: completed,
                residual_bound: residual,
                converged: true,
            };
        }

        if beta <= f64::EPSILON * scale {
            return LanczosEstimate {
                largest_eigenvalue: from_f64::<Item>(best_value),
                iterations: completed,
                residual_bound: residual,
                converged: true,
            };
        }

        previous_beta = beta;
        off_diagonal.push(beta);
        w.scale_inplace(Item::from_real(from_f64::<Item>(1.0 / beta)));
        q = w;
    }

    LanczosEstimate {
        largest_eigenvalue: from_f64::<Item>(best_value),
        iterations: options.max_iterations,
        residual_bound: best_residual,
        converged: false,
    }
}

fn zero_estimate<Item: RlstScalar>() -> LanczosEstimate<Real<Item>> {
    LanczosEstimate {
        largest_eigenvalue: from_f64::<Item>(0.0),
        iterations: 0,
        residual_bound: 0.0,
        converged: true,
    }
}

fn as_f64<T: NumCast>(value: T) -> f64 {
    NumCast::from(value).expect("Lanczos scalar must convert to f64")
}

fn from_f64<Item: RlstScalar>(value: f64) -> Real<Item> {
    NumCast::from(value).expect("f64 must convert to the Lanczos real scalar")
}

/// Largest eigenpair of a small real symmetric tridiagonal matrix.
/// A Jacobi eigensolver is adequate here because the matrix has at most a few
/// dozen rows and no expensive operator applications are involved.
fn largest_ritz_pair(diagonal: &[f64], off_diagonal: &[f64]) -> (f64, f64) {
    let n = diagonal.len();
    let mut matrix = vec![vec![0.0; n]; n];
    let mut vectors = vec![vec![0.0; n]; n];
    for i in 0..n {
        matrix[i][i] = diagonal[i];
        vectors[i][i] = 1.0;
        if i + 1 < n {
            matrix[i][i + 1] = off_diagonal[i];
            matrix[i + 1][i] = off_diagonal[i];
        }
    }

    let tolerance = 1.0e-14;
    for _ in 0..(50 * n.max(1) * n.max(1)) {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                if matrix[i][j].abs() > largest {
                    largest = matrix[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if largest <= tolerance {
            break;
        }

        let angle = 0.5 * (2.0 * matrix[p][q]).atan2(matrix[q][q] - matrix[p][p]);
        let (s, c) = angle.sin_cos();
        for k in 0..n {
            if k != p && k != q {
                let mkp = matrix[k][p];
                let mkq = matrix[k][q];
                matrix[k][p] = c * mkp - s * mkq;
                matrix[p][k] = matrix[k][p];
                matrix[k][q] = s * mkp + c * mkq;
                matrix[q][k] = matrix[k][q];
            }
        }
        let app = matrix[p][p];
        let aqq = matrix[q][q];
        let apq = matrix[p][q];
        matrix[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p][q] = 0.0;
        matrix[q][p] = 0.0;
        for row in &mut vectors {
            let vip = row[p];
            let viq = row[q];
            row[p] = c * vip - s * viq;
            row[q] = s * vip + c * viq;
        }
    }

    let index = (0..n)
        .max_by(|&i, &j| matrix[i][i].partial_cmp(&matrix[j][j]).unwrap())
        .unwrap();
    (matrix[index][index], vectors[n - 1][index])
}

#[cfg(test)]
mod tests {
    use super::largest_ritz_pair;

    #[test]
    fn ritz_solver_handles_known_tridiagonal() {
        // Eigenvalues are 2 + 2 cos(j*pi/4); the largest is 2 + sqrt(2).
        let (value, last_component) = largest_ritz_pair(&[2.0, 2.0, 2.0], &[1.0, 1.0]);
        assert!((value - (2.0 + 2.0_f64.sqrt())).abs() < 1.0e-12);
        assert!(last_component.abs() > 0.0);
    }
}
