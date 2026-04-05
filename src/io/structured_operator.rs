use crate::io::structured_operators_types::StructuredOperatorType;
use crate::test_prep::Precision;
use bempp_octree::Point;
use num::Complex;
use rlst::dense::linalg::lu::MatrixLu;
use rlst::dense::tools::RandScalar;
use rlst::operator::ConcreteElementContainer;
use rlst::prelude::*;
use rlst::RlstScalar;
use serde::Deserialize;
use std::ffi::CString;
use std::rc::Rc;
use std::time::Instant;

#[repr(C)]
struct StructuredOperatorOpaque {
    _private: [u8; 0],
}

extern "C" {
    // -------------------------
    // Initialization / finalization
    // -------------------------
    fn initialize_structured_operator(
        python_executable: *const std::ffi::c_char,
        class_name: *const std::ffi::c_char,
        arg1: libc::c_double,
        geometry_type: *const std::ffi::c_char,
        kappa: libc::c_double,
        precision: *const std::ffi::c_char,
        n_sources: std::ffi::c_int,
        init_samples: std::ffi::c_int,
        assembler: *const std::ffi::c_char,
    ) -> *mut StructuredOperatorOpaque;

    fn finalize_structured_operator(structured_operator: *mut StructuredOperatorOpaque);

    // -------------------------
    // Matrix-vector multiplication
    // -------------------------
    fn mv_structured_operator_real(
        structured_operator: *mut StructuredOperatorOpaque,
        input: *const f64,
        output: *mut f64,
        len: libc::c_int,
    ) -> libc::c_int;

    fn mv_structured_operator_complex(
        structured_operator: *mut StructuredOperatorOpaque,
        input: *const num::Complex<f64>,
        output: *mut num::Complex<f64>,
        len: libc::c_int,
    ) -> libc::c_int;

    fn mv_structured_operator_real_trans(
        structured_operator: *mut StructuredOperatorOpaque,
        input: *const f64,
        output: *mut f64,
        len: libc::c_int,
    ) -> libc::c_int;

    fn mv_structured_operator_complex_trans(
        structured_operator: *mut StructuredOperatorOpaque,
        input: *const num::Complex<f64>,
        output: *mut num::Complex<f64>,
        len: libc::c_int,
    ) -> libc::c_int;

    fn mv_structured_operator_real32_trans(
        op: *mut StructuredOperatorOpaque,
        input: *const f32,
        output: *mut f32,
        len: libc::c_int,
    ) -> libc::c_int;

    fn mv_structured_operator_complex32_trans(
        op: *mut StructuredOperatorOpaque,
        input: *const num::Complex<f32>,
        output: *mut num::Complex<f32>,
        len: libc::c_int,
    ) -> libc::c_int;

    // -------------------------
    // Geometry / info
    // -------------------------
    fn get_points(structured_operator: *mut StructuredOperatorOpaque) -> *const f64;
    fn get_n_points(structured_operator: *mut StructuredOperatorOpaque) -> usize;

    // -------------------------
    // Multiple RHS retrieval
    // -------------------------
    fn get_all_real_rhs(
        structured_operator: *mut StructuredOperatorOpaque,
        n_rhs: *mut libc::c_int,
        len_out: *mut libc::c_int,
    ) -> *const *const f64;

    fn get_all_complex_rhs(
        structured_operator: *mut StructuredOperatorOpaque,
        n_rhs: *mut libc::c_int,
        len_out: *mut libc::c_int,
    ) -> *const *const num::Complex<f64>;

    fn mv_structured_operator_real32(
        op: *mut StructuredOperatorOpaque,
        input: *const f32,
        output: *mut f32,
        len: libc::c_int,
    ) -> libc::c_int;

    fn mv_structured_operator_complex32(
        op: *mut StructuredOperatorOpaque,
        input: *const num::Complex<f32>,
        output: *mut num::Complex<f32>,
        len: libc::c_int,
    ) -> libc::c_int;

    fn get_all_real_rhs_f32(
        op: *mut StructuredOperatorOpaque,
        n_rhs: *mut libc::c_int,
        len_out: *mut libc::c_int,
    ) -> *const *const f32;

    fn get_all_complex_rhs_f32(
        op: *mut StructuredOperatorOpaque,
        n_rhs: *mut libc::c_int,
        len_out: *mut libc::c_int,
    ) -> *const *const num::Complex<f32>;
}

#[derive(Clone)]
pub struct StructuredOperatorInterface {
    raw: *mut StructuredOperatorOpaque,
    pub n_points: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub enum GeometryType {
    SphereSurface,
    CubeSurface,
    CylinderSurface,
    EllipsoidSurface,
    TrefoilKnot,
    Sphere,
    Cube,
    Dihedral,
    Device,
    F16,
    RidgedHorn,
    EMCCAlmond,
    FrigateHull,
    Plane,
    Square,
}

#[derive(Debug, Clone, Deserialize)]
pub enum Assembler {
    Dense,
    FMM,
}

pub struct StructuredOperatorParams {
    pub structured_operator_type: StructuredOperatorType,
    precision: Precision,
    geometry_type: GeometryType,
    dim_arg: f64,
    kappa: f64,
    n_sources: i32,
    init_samples: i32,
    assembler: Assembler,
    sample_storage_dir: Option<String>,
}
impl StructuredOperatorParams {
    pub fn new(
        structured_operator_type: StructuredOperatorType,
        precision: Precision,
        geometry_type: GeometryType,
        dim_arg: f64,
        kappa: f64,
        n_sources: i32,
        init_samples: i32,
        assembler: Assembler,
    ) -> Self {
        Self {
            structured_operator_type,
            precision,
            geometry_type,
            dim_arg,
            kappa,
            n_sources,
            init_samples,
            assembler,
            sample_storage_dir: None,
        }
    }

    pub fn with_sample_storage_dir(mut self, sample_storage_dir: impl Into<String>) -> Self {
        self.sample_storage_dir = Some(sample_storage_dir.into());
        self
    }
}
/*type BemppRsOperator<T> = Operator<T>;
pub enum StructuredOperatorImplType<T: OperatorBase + AsApply> {
    Python(StructuredOperatorInterface),
    Rust(BemppRsOperator<T>),
}*/

pub trait StructuredOperatorImpl<Item: RlstScalar> {
    type Item: RlstScalar;
    fn new(params: &StructuredOperatorParams) -> Self;
    fn mv(&self, input: &[Item], output: &mut [Item]);
    fn mv_trans(&self, input: &[Item], output: &mut [Item]);
    fn get_points(&self) -> Option<Vec<Point>>;
    fn rhs(&self) -> Vec<Vec<Self::Item>>; // updated to multi-RHS
}

fn query_python_env(python_executable: &str, code: &str) -> (String, String) {
    let output = std::process::Command::new(python_executable)
        .arg("-c")
        .arg(code)
        .output()
        .expect("Failed to run Python");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut lines = stdout.lines();
    let executable = lines.next().unwrap_or("").to_string();
    let prefix = lines.next().unwrap_or("").to_string();
    (executable, prefix)
}

fn detect_python_env() -> (String, String) {
    let code = r#"
import sys
print(sys.executable)
print(sys.prefix)
"#;

    if let Ok(python_executable) = std::env::var("RSRS_EXPS_PYTHON") {
        return query_python_env(&python_executable, code);
    }

    if let Ok(version_text) = std::fs::read_to_string(".python-version") {
        if let Some(version) = version_text
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
        {
            if let Ok(output) = std::process::Command::new("pyenv")
                .env("PYENV_VERSION", version)
                .args(["which", "python"])
                .output()
            {
                if output.status.success() {
                    let python_executable =
                        String::from_utf8_lossy(&output.stdout).trim().to_string();
                    if !python_executable.is_empty() {
                        return query_python_env(&python_executable, code);
                    }
                }
            }
        }
    }

    query_python_env("python3", code)
}

macro_rules! implement_structured_operator {
    ($scalar:ty, $mv:expr, $mv_t:expr, $rhs_fn:expr) => {
        impl StructuredOperatorImpl<$scalar> for StructuredOperatorInterface {
            type Item = $scalar;

            fn new(params: &StructuredOperatorParams) -> Self {
                if let Some(sample_storage_dir) = params.sample_storage_dir.as_deref() {
                    std::env::set_var("RSRS_SAMPLE_STORAGE_DIR", sample_storage_dir);
                } else {
                    std::env::remove_var("RSRS_SAMPLE_STORAGE_DIR");
                }

                let class_name = params.structured_operator_type.as_ref();
                let c_str = std::ffi::CString::new(class_name).unwrap();
                let precision_str = match params.precision {
                    Precision::Double => std::ffi::CString::new("double").unwrap(),
                    Precision::Single => std::ffi::CString::new("single").unwrap(),
                };

                let geometry = std::ffi::CString::new(match params.geometry_type {
                    GeometryType::SphereSurface => "sphere_surface",
                    GeometryType::CubeSurface => "cube_surface",
                    GeometryType::CylinderSurface => "cylinder_surface",
                    GeometryType::EllipsoidSurface => "ellipsoid_surface",
                    GeometryType::TrefoilKnot => "trefoil_knot",
                    GeometryType::Sphere => "sphere",
                    GeometryType::Cube => "cube",
                    GeometryType::Dihedral => "dihedral",
                    GeometryType::Device => "device",
                    GeometryType::F16 => "f16",
                    GeometryType::RidgedHorn => "ridged_horn",
                    GeometryType::EMCCAlmond => "emcc_almond",
                    GeometryType::FrigateHull => "frigate_hull",
                    GeometryType::Plane => "plane",
                    GeometryType::Square => "square",
                })
                .unwrap();

                let assembler = std::ffi::CString::new(match params.assembler {
                    Assembler::Dense => "dense",
                    Assembler::FMM => "fmm",
                })
                .unwrap();

                println!(
                    "[rsrs-exps][structured-operator] initialize start: class='{}', geometry='{}', dim_arg={:.6e}, kappa={:.6e}, init_samples={}, sample_dir={}",
                    class_name,
                    geometry.to_string_lossy(),
                    params.dim_arg,
                    params.kappa,
                    params.init_samples,
                    params.sample_storage_dir.as_deref().unwrap_or("<none>")
                );

                let env_stage = Instant::now();
                let (python_executable, _python_home) = detect_python_env();
                println!(
                    "[rsrs-exps][structured-operator] python environment resolved in {:.3}s: executable='{}'",
                    env_stage.elapsed().as_secs_f64(),
                    python_executable
                );
                let python_exe_c = CString::new(python_executable).unwrap();
                let init_stage = Instant::now();
                let raw = unsafe {
                    initialize_structured_operator(
                        python_exe_c.as_ptr(),
                        c_str.as_ptr(),
                        params.dim_arg as f64,
                        geometry.as_ptr(),
                        params.kappa,
                        precision_str.as_ptr(),
                        params.n_sources as libc::c_int,
                        params.init_samples as libc::c_int,
                        assembler.as_ptr(),
                    )
                };

                assert!(!raw.is_null(), "Failed to initialize structured_operator");
                let n_points = unsafe { get_n_points(raw as *mut StructuredOperatorOpaque) };
                println!(
                    "[rsrs-exps][structured-operator] initialize done in {:.3}s: n_points={}",
                    init_stage.elapsed().as_secs_f64(),
                    n_points,
                );
                Self { raw, n_points }
            }

            fn mv(&self, input: &[Self::Item], output: &mut [Self::Item]) {
                assert_eq!(input.len(), self.n_points);
                assert_eq!(output.len(), self.n_points);
                unsafe {
                    assert!(
                        $mv(
                            self.raw,
                            input.as_ptr(),
                            output.as_mut_ptr(),
                            self.n_points as libc::c_int
                        ) != 0,
                        "mv_structured_operator call failed"
                    );
                }
            }

            fn mv_trans(&self, input: &[Self::Item], output: &mut [Self::Item]) {
                unsafe {
                    assert!(
                        $mv_t(
                            self.raw,
                            input.as_ptr(),
                            output.as_mut_ptr(),
                            self.n_points as libc::c_int
                        ) != 0
                    );
                }
            }

            fn rhs(&self) -> Vec<Vec<$scalar>> {
                $rhs_fn(&self).unwrap()
            }

            fn get_points(&self) -> Option<Vec<Point>> {
                get_bempp_points(&self)
            }
        }
    };
}

// -------------------------
// Multi-RHS safe Rust wrappers
// -------------------------
pub fn rhs_real(structured_operator: &StructuredOperatorInterface) -> Option<Vec<Vec<f64>>> {
    let stage = Instant::now();
    println!(
        "[rsrs-exps][structured-operator] fetch real rhs start: n_points={}",
        structured_operator.n_points
    );
    let mut n_rhs: libc::c_int = 0;
    let mut len_out: libc::c_int = 0;

    let ptr = unsafe { get_all_real_rhs(structured_operator.raw, &mut n_rhs, &mut len_out) };
    if ptr.is_null() || n_rhs <= 0 || len_out <= 0 {
        return None;
    }

    let slice = unsafe { std::slice::from_raw_parts(ptr, n_rhs as usize) };
    let mut all_rhs = Vec::with_capacity(n_rhs as usize);

    for &rhs_ptr in slice.iter() {
        if rhs_ptr.is_null() {
            unsafe { libc::free(ptr as *mut libc::c_void) };
            return None;
        }
        let rhs_slice = unsafe { std::slice::from_raw_parts(rhs_ptr, len_out as usize) };
        all_rhs.push(rhs_slice.to_vec());
    }

    unsafe { libc::free(ptr as *mut libc::c_void) };
    println!(
        "[rsrs-exps][structured-operator] fetch real rhs done in {:.3}s: rhs_count={}, rhs_len={}",
        stage.elapsed().as_secs_f64(),
        all_rhs.len(),
        len_out
    );
    Some(all_rhs)
}

pub fn rhs_real32(structured_operator: &StructuredOperatorInterface) -> Option<Vec<Vec<f32>>> {
    let stage = Instant::now();
    println!(
        "[rsrs-exps][structured-operator] fetch real32 rhs start: n_points={}",
        structured_operator.n_points
    );
    let mut n_rhs: libc::c_int = 0;
    let mut len_out: libc::c_int = 0;

    let ptr = unsafe { get_all_real_rhs_f32(structured_operator.raw, &mut n_rhs, &mut len_out) };
    if ptr.is_null() || n_rhs <= 0 || len_out <= 0 {
        return None;
    }

    let slice = unsafe { std::slice::from_raw_parts(ptr, n_rhs as usize) };
    let mut all_rhs = Vec::with_capacity(n_rhs as usize);

    for &rhs_ptr in slice.iter() {
        if rhs_ptr.is_null() {
            unsafe { libc::free(ptr as *mut libc::c_void) };
            return None;
        }
        let rhs_slice = unsafe { std::slice::from_raw_parts(rhs_ptr, len_out as usize) };
        all_rhs.push(rhs_slice.to_vec());
    }

    unsafe { libc::free(ptr as *mut libc::c_void) };
    println!(
        "[rsrs-exps][structured-operator] fetch real32 rhs done in {:.3}s: rhs_count={}, rhs_len={}",
        stage.elapsed().as_secs_f64(),
        all_rhs.len(),
        len_out
    );
    Some(all_rhs)
}

pub fn rhs_complex(
    structured_operator: &StructuredOperatorInterface,
) -> Option<Vec<Vec<num::Complex<f64>>>> {
    let stage = Instant::now();
    println!(
        "[rsrs-exps][structured-operator] fetch complex rhs start: n_points={}",
        structured_operator.n_points
    );
    let mut n_rhs: libc::c_int = 0;
    let mut len_out: libc::c_int = 0;

    let ptr = unsafe { get_all_complex_rhs(structured_operator.raw, &mut n_rhs, &mut len_out) };
    if ptr.is_null() || n_rhs <= 0 || len_out <= 0 {
        return None;
    }

    let slice = unsafe { std::slice::from_raw_parts(ptr, n_rhs as usize) };
    let mut all_rhs = Vec::with_capacity(n_rhs as usize);

    for &rhs_ptr in slice.iter() {
        if rhs_ptr.is_null() {
            unsafe { libc::free(ptr as *mut libc::c_void) };
            return None;
        }
        let rhs_slice = unsafe { std::slice::from_raw_parts(rhs_ptr, len_out as usize) };
        all_rhs.push(rhs_slice.to_vec());
    }

    unsafe { libc::free(ptr as *mut libc::c_void) };
    println!(
        "[rsrs-exps][structured-operator] fetch complex rhs done in {:.3}s: rhs_count={}, rhs_len={}",
        stage.elapsed().as_secs_f64(),
        all_rhs.len(),
        len_out
    );
    Some(all_rhs)
}

pub fn rhs_complex32(
    structured_operator: &StructuredOperatorInterface,
) -> Option<Vec<Vec<num::Complex<f32>>>> {
    let stage = Instant::now();
    println!(
        "[rsrs-exps][structured-operator] fetch complex32 rhs start: n_points={}",
        structured_operator.n_points
    );
    let mut n_rhs: libc::c_int = 0;
    let mut len_out: libc::c_int = 0;

    let ptr = unsafe { get_all_complex_rhs_f32(structured_operator.raw, &mut n_rhs, &mut len_out) };
    if ptr.is_null() || n_rhs <= 0 || len_out <= 0 {
        return None;
    }

    let slice = unsafe { std::slice::from_raw_parts(ptr, n_rhs as usize) };
    let mut all_rhs = Vec::with_capacity(n_rhs as usize);

    for &rhs_ptr in slice.iter() {
        if rhs_ptr.is_null() {
            unsafe { libc::free(ptr as *mut libc::c_void) };
            return None;
        }
        let rhs_slice = unsafe { std::slice::from_raw_parts(rhs_ptr, len_out as usize) };
        all_rhs.push(rhs_slice.to_vec());
    }

    unsafe { libc::free(ptr as *mut libc::c_void) };
    println!(
        "[rsrs-exps][structured-operator] fetch complex32 rhs done in {:.3}s: rhs_count={}, rhs_len={}",
        stage.elapsed().as_secs_f64(),
        all_rhs.len(),
        len_out
    );
    Some(all_rhs)
}

implement_structured_operator!(
    f32,
    mv_structured_operator_real32,
    mv_structured_operator_real32_trans,
    rhs_real32
);
implement_structured_operator!(
    f64,
    mv_structured_operator_real,
    mv_structured_operator_real_trans,
    rhs_real
);
implement_structured_operator!(
    Complex<f32>,
    mv_structured_operator_complex32,
    mv_structured_operator_complex32_trans,
    rhs_complex32
);
implement_structured_operator!(
    Complex<f64>,
    mv_structured_operator_complex,
    mv_structured_operator_complex_trans,
    rhs_complex
);

// -------------------------
// Convert 1D points array to Bempp Points
// -------------------------
pub fn one_dim_to_bempp_points(raw_points: &[f64]) -> Vec<bempp_octree::Point> {
    raw_points
        .chunks_exact(3)
        .map(|chunk| bempp_octree::Point::new([chunk[0], chunk[1], chunk[2]], 0))
        .collect()
}

pub fn get_bempp_points(structured_operator: &StructuredOperatorInterface) -> Option<Vec<Point>> {
    let ptr = unsafe { get_points(structured_operator.raw) };
    if ptr.is_null() {
        return None;
    }
    let total_len = structured_operator.n_points * 3;
    let slice = unsafe { std::slice::from_raw_parts(ptr, total_len) };

    Some({
        let raw_points = unsafe {
            std::slice::from_raw_parts(
                slice.as_ptr() as *const [f64; 3],
                structured_operator.n_points,
            )
        };

        raw_points.iter().map(|&el| Point::new(el, 0)).collect()
    })
}

impl Drop for StructuredOperatorInterface {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { finalize_structured_operator(self.raw) }
            self.raw = std::ptr::null_mut();
        }
    }
}

// -------------------------
// Remaining operator traits and wrappers (unchanged)
// -------------------------
impl Shape<2> for StructuredOperatorInterface {
    fn shape(&self) -> [usize; 2] {
        [self.n_points, self.n_points]
    }
}

#[derive(Clone)]
pub struct StructuredOperator<Item: RlstScalar, Op: StructuredOperatorImpl<Item> + Shape<2>> {
    op: Op,
    domain: Rc<ArrayVectorSpace<Item>>,
    range: Rc<ArrayVectorSpace<Item>>,
}

pub trait Attr<Op, Item: RlstScalar>: Sized {
    /// Returns a vector of `Element`s, one for each RHS.
    fn get_rhs(&self) -> Vec<Element<ConcreteElementContainer<ArrayVectorSpaceElement<Item>>>>;
}

impl<Item: RlstScalar, Op: StructuredOperatorImpl<Item> + Shape<2>> Attr<Op, Item>
    for StructuredOperator<Item, Op>
where
    Op: StructuredOperatorImpl<Item, Item = Item>,
{
    fn get_rhs(&self) -> Vec<Element<ConcreteElementContainer<ArrayVectorSpaceElement<Item>>>> {
        let all_rhs = self.op.rhs(); // Vec<Vec<Item>>, each inner Vec is one RHS
        let mut elements = Vec::with_capacity(all_rhs.len());

        for rhs_vec in all_rhs.into_iter() {
            let mut elem: Element<ConcreteElementContainer<ArrayVectorSpaceElement<Item>>> =
                zero_element(self.domain.clone());
            for (i, val) in elem.imp_mut().view_mut().data_mut().iter_mut().enumerate() {
                *val = rhs_vec[i];
            }
            elements.push(elem);
        }

        elements
    }
}

impl<Item: RlstScalar, Op: StructuredOperatorImpl<Item> + Shape<2>> std::fmt::Debug
    for StructuredOperator<Item, Op>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.op.shape();
        write!(f, "StructuredOperator: [{}x{}]", shape[0], shape[1])
    }
}

pub struct LocalOp<'a, Op> {
    pub op: &'a Op,
}

impl<Item: RlstScalar, Op: StructuredOperatorImpl<Item> + Shape<2>> OperatorBase
    for StructuredOperator<Item, Op>
{
    type Domain = ArrayVectorSpace<Item>;
    type Range = ArrayVectorSpace<Item>;

    fn domain(&self) -> Rc<Self::Domain> {
        self.domain.clone()
    }

    fn range(&self) -> Rc<Self::Range> {
        self.range.clone()
    }
}

pub trait LocalFrom<Op, Item: RlstScalar>: Sized {
    fn from_local(op: Op) -> Self;
}

impl<
        'a,
        Item: RlstScalar
            + MatrixInverse
            + MatrixId
            + MatrixPseudoInverse
            + MatrixLu
            + RandScalar
            + MatrixQr,
        Op: StructuredOperatorImpl<Item> + Shape<2>,
    > LocalFrom<Op, Item> for StructuredOperator<Item, Op>
{
    fn from_local(op: Op) -> Self {
        let shape = op.shape();
        let domain = ArrayVectorSpace::from_dimension(shape[1]);
        let range = ArrayVectorSpace::from_dimension(shape[0]);
        StructuredOperator { op, domain, range }
    }
}

impl<Item: RlstScalar, Op: StructuredOperatorImpl<Item> + Shape<2>> AsApply
    for StructuredOperator<Item, Op>
{
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
        trans_mode: TransMode,
    ) {
        match trans_mode {
            TransMode::NoTrans => {
                self.op
                    .mv(x.imp().view().data(), y.imp_mut().view_mut().data_mut());
            }
            TransMode::Trans => {
                self.op
                    .mv_trans(x.imp().view().data(), y.imp_mut().view_mut().data_mut());
            }
            TransMode::ConjNoTrans | TransMode::ConjTrans => {
                panic!("Conjugate transpose modes not supported for multiplication.")
            }
        }
    }
}
