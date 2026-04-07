use num::Complex;
use rlst::prelude::*;
use rsrs_exps::{
    io::{
        structured_operator::{
            Assembler, GeometryType, LocalFrom, StructuredOperator, StructuredOperatorImpl,
            StructuredOperatorInterface, StructuredOperatorParams,
        },
        structured_operators_types::StructuredOperatorType,
    },
    test_prep::Precision,
};

type Item = Complex<f64>;
type RealItem = f64;

fn basis_vector(dim: usize, index: usize) -> Vec<Item> {
    let mut basis = vec![Item::new(0.0, 0.0); dim];
    basis[index] = Item::new(1.0, 0.0);
    basis
}

fn basis_vector_real(dim: usize, index: usize) -> Vec<RealItem> {
    let mut basis = vec![0.0; dim];
    basis[index] = 1.0;
    basis
}

fn apply_operator<Op>(op: &Op, input: &[Item], trans_mode: TransMode) -> Vec<Item>
where
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
{
    let mut x = zero_element(op.domain());
    x.imp_mut().fill_inplace_raw(input);
    let y = op.apply(x.r(), trans_mode);
    y.view().iter().collect()
}

fn apply_operator_real<Op>(op: &Op, input: &[RealItem], trans_mode: TransMode) -> Vec<RealItem>
where
    Op: AsApply<Domain = ArrayVectorSpace<RealItem>, Range = ArrayVectorSpace<RealItem>>,
{
    let mut x = zero_element(op.domain());
    x.imp_mut().fill_inplace_raw(input);
    let y = op.apply(x.r(), trans_mode);
    y.view().iter().collect()
}

fn assemble_matrix<Op>(op: &Op, dim: usize, trans_mode: TransMode) -> DynamicArray<Item, 2>
where
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
{
    let mut matrix = rlst_dynamic_array2!(Item, [dim, dim]);

    for col in 0..dim {
        let basis = basis_vector(dim, col);
        let column = apply_operator(op, &basis, trans_mode);
        for (row, value) in column.into_iter().enumerate() {
            matrix[[row, col]] = value;
        }
    }

    matrix
}

fn assemble_matrix_real<Op>(op: &Op, dim: usize, trans_mode: TransMode) -> DynamicArray<RealItem, 2>
where
    Op: AsApply<Domain = ArrayVectorSpace<RealItem>, Range = ArrayVectorSpace<RealItem>>,
{
    let mut matrix = rlst_dynamic_array2!(RealItem, [dim, dim]);

    for col in 0..dim {
        let basis = basis_vector_real(dim, col);
        let column = apply_operator_real(op, &basis, trans_mode);
        for (row, value) in column.into_iter().enumerate() {
            matrix[[row, col]] = value;
        }
    }

    matrix
}

fn adjoint(matrix: &DynamicArray<Item, 2>) -> DynamicArray<Item, 2> {
    let shape = matrix.shape();
    let mut out = rlst_dynamic_array2!(Item, [shape[1], shape[0]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[col, row]] = matrix[[row, col]].conj();
        }
    }
    out
}

fn diff(lhs: &DynamicArray<Item, 2>, rhs: &DynamicArray<Item, 2>) -> DynamicArray<Item, 2> {
    let shape = lhs.shape();
    let mut out = rlst_dynamic_array2!(Item, [shape[0], shape[1]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[row, col]] = lhs[[row, col]] - rhs[[row, col]];
        }
    }
    out
}

fn transpose_real(matrix: &DynamicArray<RealItem, 2>) -> DynamicArray<RealItem, 2> {
    let shape = matrix.shape();
    let mut out = rlst_dynamic_array2!(RealItem, [shape[1], shape[0]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[col, row]] = matrix[[row, col]];
        }
    }
    out
}

fn diff_real(
    lhs: &DynamicArray<RealItem, 2>,
    rhs: &DynamicArray<RealItem, 2>,
) -> DynamicArray<RealItem, 2> {
    let shape = lhs.shape();
    let mut out = rlst_dynamic_array2!(RealItem, [shape[0], shape[1]]);
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            out[[row, col]] = lhs[[row, col]] - rhs[[row, col]];
        }
    }
    out
}

fn frobenius_norm(matrix: &DynamicArray<Item, 2>) -> f64 {
    matrix
        .r()
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn imag_frobenius_norm(matrix: &DynamicArray<Item, 2>) -> f64 {
    matrix
        .r()
        .iter()
        .map(|value| value.im * value.im)
        .sum::<f64>()
        .sqrt()
}

fn frobenius_norm_real(matrix: &DynamicArray<RealItem, 2>) -> f64 {
    matrix
        .r()
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn complex_test_vector(dim: usize, offset: f64) -> Vec<Item> {
    (0..dim)
        .map(|idx| {
            let t = idx as f64 + 1.0 + offset;
            Item::new((0.31 * t).sin() + 0.2 * (0.13 * t).cos(), (0.17 * t).cos())
        })
        .collect()
}

fn real_test_vector(dim: usize, offset: f64) -> Vec<RealItem> {
    (0..dim)
        .map(|idx| {
            let t = idx as f64 + 1.0 + offset;
            (0.29 * t).sin() + 0.25 * (0.11 * t).cos()
        })
        .collect()
}

fn complex_inner(lhs: &[Item], rhs: &[Item]) -> Item {
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l.conj() * *r).sum()
}

fn real_inner(lhs: &[RealItem], rhs: &[RealItem]) -> RealItem {
    lhs.iter().zip(rhs.iter()).map(|(l, r)| l * r).sum()
}

fn relative_complex_adjoint_defect<Op>(op: &Op, dim: usize) -> f64
where
    Op: AsApply<Domain = ArrayVectorSpace<Item>, Range = ArrayVectorSpace<Item>>,
{
    let x = complex_test_vector(dim, 0.0);
    let y = complex_test_vector(dim, 0.5);
    let ay = apply_operator(op, &y, TransMode::NoTrans);
    let ahx = apply_operator(op, &x, TransMode::Trans);
    let left = complex_inner(&x, &ay);
    let right = complex_inner(&ahx, &y);
    (left - right).norm() / left.norm().max(right.norm()).max(1.0e-14)
}

fn relative_real_adjoint_defect<Op>(op: &Op, dim: usize) -> f64
where
    Op: AsApply<Domain = ArrayVectorSpace<RealItem>, Range = ArrayVectorSpace<RealItem>>,
{
    let x = real_test_vector(dim, 0.0);
    let y = real_test_vector(dim, 0.5);
    let ay = apply_operator_real(op, &y, TransMode::NoTrans);
    let atx = apply_operator_real(op, &x, TransMode::Trans);
    let left = real_inner(&x, &ay);
    let right = real_inner(&atx, &y);
    (left - right).abs() / left.abs().max(right.abs()).max(1.0e-14)
}

fn run_biegrid_apply_and_transpose_consistency_check() {
    let params = StructuredOperatorParams::new(
        StructuredOperatorType::BIEGrid,
        Precision::Double,
        GeometryType::Square,
        0.05,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface = <StructuredOperatorInterface as StructuredOperatorImpl<RealItem>>::new(&params);
    let operator =
        StructuredOperator::<RealItem, StructuredOperatorInterface>::from_local(interface);

    let dim = operator.domain().dimension();
    let matrix = assemble_matrix_real(&operator, dim, TransMode::NoTrans);
    let matrix_trans_route = assemble_matrix_real(&operator, dim, TransMode::Trans);
    let exact_transpose = transpose_real(&matrix);

    let rel_transpose_route =
        frobenius_norm_real(&diff_real(&matrix_trans_route, &exact_transpose))
            / frobenius_norm_real(&exact_transpose).max(1.0e-14);
    let rel_symmetry = frobenius_norm_real(&diff_real(&matrix, &exact_transpose))
        / frobenius_norm_real(&matrix).max(1.0e-14);
    let rel_apply_adjoint = relative_real_adjoint_defect(&operator, dim);

    println!(
        "biegrid-real: dim={dim}, rel_transpose_route={rel_transpose_route:.3e}, rel_symmetry={rel_symmetry:.3e}, rel_apply_adjoint={rel_apply_adjoint:.3e}"
    );

    assert!(
        rel_transpose_route < 1.0e-11,
        "BIEGrid transpose route is inconsistent ({rel_transpose_route})"
    );
    assert!(
        rel_symmetry < 1.0e-11,
        "BIEGrid should be symmetric ({rel_symmetry})"
    );
    assert!(
        rel_apply_adjoint < 1.0e-11,
        "BIEGrid apply/transpose consistency is poor ({rel_apply_adjoint})"
    );
    // The FFI finalizer calls Py_Finalize(), so keep this test-only operator
    // alive while the remaining Python-backed variants are constructed.
    std::mem::forget(operator);
}

fn run_biegrid_real_perturbed_consistency_check() {
    let params = StructuredOperatorParams::new(
        StructuredOperatorType::BIEGridRealPerturbed,
        Precision::Double,
        GeometryType::Square,
        0.05,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface = <StructuredOperatorInterface as StructuredOperatorImpl<RealItem>>::new(&params);
    let operator =
        StructuredOperator::<RealItem, StructuredOperatorInterface>::from_local(interface);

    let dim = operator.domain().dimension();
    let matrix = assemble_matrix_real(&operator, dim, TransMode::NoTrans);
    let matrix_trans_route = assemble_matrix_real(&operator, dim, TransMode::Trans);
    let exact_transpose = transpose_real(&matrix);

    let rel_transpose_route =
        frobenius_norm_real(&diff_real(&matrix_trans_route, &exact_transpose))
            / frobenius_norm_real(&exact_transpose).max(1.0e-14);
    let rel_nonsymmetry = frobenius_norm_real(&diff_real(&matrix, &exact_transpose))
        / frobenius_norm_real(&matrix).max(1.0e-14);
    let rel_apply_adjoint = relative_real_adjoint_defect(&operator, dim);

    println!(
        "biegrid-real-perturbed: dim={dim}, rel_transpose_route={rel_transpose_route:.3e}, rel_nonsymmetry={rel_nonsymmetry:.3e}, rel_apply_adjoint={rel_apply_adjoint:.3e}"
    );

    assert!(
        rel_transpose_route < 1.0e-11,
        "real perturbed BIEGrid transpose route is inconsistent ({rel_transpose_route})"
    );
    assert!(
        rel_apply_adjoint < 1.0e-11,
        "real perturbed BIEGrid apply/transpose consistency is poor ({rel_apply_adjoint})"
    );
    assert!(
        rel_nonsymmetry > 1.0e-4,
        "real perturbed BIEGrid should be observably nonsymmetric ({rel_nonsymmetry})"
    );
    std::mem::forget(operator);
}

fn run_biegrid_real_symmetric_perturbed_consistency_check() {
    let params = StructuredOperatorParams::new(
        StructuredOperatorType::BIEGridRealSymmetricPerturbed,
        Precision::Double,
        GeometryType::Square,
        0.05,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface = <StructuredOperatorInterface as StructuredOperatorImpl<RealItem>>::new(&params);
    let operator =
        StructuredOperator::<RealItem, StructuredOperatorInterface>::from_local(interface);

    let dim = operator.domain().dimension();
    let matrix = assemble_matrix_real(&operator, dim, TransMode::NoTrans);
    let matrix_trans_route = assemble_matrix_real(&operator, dim, TransMode::Trans);
    let exact_transpose = transpose_real(&matrix);

    let rel_transpose_route =
        frobenius_norm_real(&diff_real(&matrix_trans_route, &exact_transpose))
            / frobenius_norm_real(&exact_transpose).max(1.0e-14);
    let rel_symmetry = frobenius_norm_real(&diff_real(&matrix, &exact_transpose))
        / frobenius_norm_real(&matrix).max(1.0e-14);
    let rel_apply_adjoint = relative_real_adjoint_defect(&operator, dim);

    println!(
        "biegrid-real-symmetric-perturbed: dim={dim}, rel_transpose_route={rel_transpose_route:.3e}, rel_symmetry={rel_symmetry:.3e}, rel_apply_adjoint={rel_apply_adjoint:.3e}"
    );

    assert!(
        rel_transpose_route < 1.0e-11,
        "real symmetric perturbed BIEGrid transpose route is inconsistent ({rel_transpose_route})"
    );
    assert!(
        rel_symmetry < 1.0e-11,
        "real symmetric perturbed BIEGrid should be symmetric ({rel_symmetry})"
    );
    assert!(
        rel_apply_adjoint < 1.0e-11,
        "real symmetric perturbed BIEGrid apply/transpose consistency is poor ({rel_apply_adjoint})"
    );
    std::mem::forget(operator);
}

fn run_biegrid_complex_perturbed_consistency_check() {
    let params = StructuredOperatorParams::new(
        StructuredOperatorType::BIEGridComplexPerturbed,
        Precision::Double,
        GeometryType::Square,
        0.05,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface = <StructuredOperatorInterface as StructuredOperatorImpl<Item>>::new(&params);
    let operator = StructuredOperator::<Item, StructuredOperatorInterface>::from_local(interface);

    let dim = operator.domain().dimension();
    let matrix = assemble_matrix(&operator, dim, TransMode::NoTrans);
    let matrix_adjoint_route = assemble_matrix(&operator, dim, TransMode::Trans);
    let exact_adjoint = adjoint(&matrix);

    let rel_adjoint_route = frobenius_norm(&diff(&matrix_adjoint_route, &exact_adjoint))
        / frobenius_norm(&exact_adjoint).max(1.0e-14);
    let rel_nonsymmetry =
        frobenius_norm(&diff(&matrix, &exact_adjoint)) / frobenius_norm(&matrix).max(1.0e-14);
    let imag_energy = imag_frobenius_norm(&matrix) / frobenius_norm(&matrix).max(1.0e-14);
    let rel_apply_adjoint = relative_complex_adjoint_defect(&operator, dim);

    println!(
        "biegrid-complex-perturbed: dim={dim}, rel_adjoint_route={rel_adjoint_route:.3e}, rel_nonsymmetry={rel_nonsymmetry:.3e}, imag_energy={imag_energy:.3e}, rel_apply_adjoint={rel_apply_adjoint:.3e}"
    );

    assert!(
        rel_adjoint_route < 1.0e-11,
        "complex perturbed BIEGrid adjoint route is inconsistent ({rel_adjoint_route})"
    );
    assert!(
        rel_nonsymmetry > 1.0e-4,
        "complex perturbed BIEGrid should be observably nonsymmetric ({rel_nonsymmetry})"
    );
    assert!(
        imag_energy > 1.0e-6,
        "complex perturbed BIEGrid should have nontrivial imaginary content ({imag_energy})"
    );
    assert!(
        rel_apply_adjoint < 1.0e-11,
        "complex perturbed BIEGrid apply/adjoint consistency is poor ({rel_apply_adjoint})"
    );
    std::mem::forget(operator);
}

fn run_biegrid_complex_symmetric_perturbed_consistency_check() {
    let params = StructuredOperatorParams::new(
        StructuredOperatorType::BIEGridComplexSymmetricPerturbed,
        Precision::Double,
        GeometryType::Square,
        0.05,
        0.0,
        1,
        0,
        Assembler::Dense,
    );

    let interface = <StructuredOperatorInterface as StructuredOperatorImpl<Item>>::new(&params);
    let operator = StructuredOperator::<Item, StructuredOperatorInterface>::from_local(interface);

    let dim = operator.domain().dimension();
    let matrix = assemble_matrix(&operator, dim, TransMode::NoTrans);
    let matrix_adjoint_route = assemble_matrix(&operator, dim, TransMode::Trans);
    let exact_adjoint = adjoint(&matrix);
    let exact_transpose = adjoint(&exact_adjoint);

    let rel_adjoint_route = frobenius_norm(&diff(&matrix_adjoint_route, &exact_adjoint))
        / frobenius_norm(&exact_adjoint).max(1.0e-14);
    let rel_transpose_symmetry =
        frobenius_norm(&diff(&matrix, &exact_transpose)) / frobenius_norm(&matrix).max(1.0e-14);
    let rel_nonhermitian =
        frobenius_norm(&diff(&matrix, &exact_adjoint)) / frobenius_norm(&matrix).max(1.0e-14);
    let imag_energy = imag_frobenius_norm(&matrix) / frobenius_norm(&matrix).max(1.0e-14);
    let rel_apply_adjoint = relative_complex_adjoint_defect(&operator, dim);

    println!(
        "biegrid-complex-symmetric-perturbed: dim={dim}, rel_adjoint_route={rel_adjoint_route:.3e}, rel_transpose_symmetry={rel_transpose_symmetry:.3e}, rel_nonhermitian={rel_nonhermitian:.3e}, imag_energy={imag_energy:.3e}, rel_apply_adjoint={rel_apply_adjoint:.3e}"
    );

    assert!(
        rel_adjoint_route < 1.0e-11,
        "complex symmetric perturbed BIEGrid adjoint route is inconsistent ({rel_adjoint_route})"
    );
    assert!(
        rel_apply_adjoint < 1.0e-11,
        "complex symmetric perturbed BIEGrid apply/adjoint consistency is poor ({rel_apply_adjoint})"
    );
    assert!(
        rel_transpose_symmetry < 1.0e-11,
        "complex symmetric perturbed BIEGrid should satisfy transpose symmetry ({rel_transpose_symmetry})"
    );
    assert!(
        rel_nonhermitian > 1.0e-4,
        "complex symmetric perturbed BIEGrid should not be Hermitian ({rel_nonhermitian})"
    );
    assert!(
        imag_energy > 1.0e-6,
        "complex symmetric perturbed BIEGrid should have nontrivial imaginary content ({imag_energy})"
    );
    std::mem::forget(operator);
}

#[test]
fn biegrid_apply_and_adjoint_variants_are_consistent() {
    run_biegrid_apply_and_transpose_consistency_check();
    run_biegrid_real_symmetric_perturbed_consistency_check();
    run_biegrid_real_perturbed_consistency_check();
    run_biegrid_complex_perturbed_consistency_check();
    run_biegrid_complex_symmetric_perturbed_consistency_check();
}
