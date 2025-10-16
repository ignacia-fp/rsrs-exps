//use crate::test_prep::DimArg;
use crate::io::structured_operators_types::StructuredOperatorType;
use crate::test_prep::Precision;
use bempp_octree::Point;
use rlst::dense::linalg::lu::MatrixLu;
use rlst::dense::tools::RandScalar;
use rlst::operator::ConcreteElementContainer;
use rlst::prelude::*;
use rlst::RlstScalar;
use serde::Deserialize;
use std::ffi::CString;
use std::rc::Rc;

#[repr(C)]
struct StructuredOperatorOpaque {
    _private: [u8; 0],
}

extern "C" {
    fn initialize_structured_operator(
        python_executable: *const std::ffi::c_char,
        class_name: *const std::ffi::c_char,
        arg1: libc::c_double,
        geometry_type: *const std::ffi::c_char,
        kappa: libc::c_double,
        precision: *const std::ffi::c_char,
    ) -> *mut StructuredOperatorOpaque;
    fn mv_structured_operator_real(
        structured_operator: *mut StructuredOperatorOpaque,
        input: *const f64,
        output: *mut f64,
        len: libc::c_int,
    ) -> libc::c_int;
    fn mv_structured_operator_complex(
        structured_operator: *mut StructuredOperatorOpaque,
        input: *const num::Complex<f64>, // or *const libc::c_void
        output: *mut num::Complex<f64>,
        len: libc::c_int,
    ) -> libc::c_int;
    fn structured_operator_get_real_rhs(
        structured_operator: *mut StructuredOperatorOpaque,
    ) -> *const f64;
    fn structured_operator_get_complex_rhs(
        structured_operator: *mut StructuredOperatorOpaque,
    ) -> *const c64;
    fn get_points(structured_operator: *mut StructuredOperatorOpaque) -> *const f64;
    fn get_condition_number(structured_operator: *mut StructuredOperatorOpaque) -> f64;
    fn get_n_points(structured_operator: *mut StructuredOperatorOpaque) -> usize;
    fn finalize_structured_operator(structured_operator: *mut StructuredOperatorOpaque);
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
    Satellite1,
}

pub struct StructuredOperatorParams {
    pub structured_operator_type: StructuredOperatorType,
    precision: Precision,
    geometry_type: GeometryType,
    dim_arg: f64,
    kappa: f64,
}

impl StructuredOperatorParams {
    pub fn new(
        structured_operator_type: StructuredOperatorType,
        precision: Precision,
        geometry_type: GeometryType,
        dim_arg: f64,
        kappa: f64,
    ) -> Self {
        Self {
            structured_operator_type,
            precision,
            geometry_type,
            dim_arg,
            kappa,
        }
    }
}

type BemppRsOperator<T> = Operator<T>;
pub enum StructuredOperatorImplType<T: OperatorBase + AsApply> {
    Python(StructuredOperatorInterface),
    Rust(BemppRsOperator<T>),
}

type Real<T> = <T as rlst::RlstScalar>::Real;
pub trait StructuredOperatorImpl<Item: RlstScalar> {
    type Item: RlstScalar;
    fn new(params: &StructuredOperatorParams) -> Self;
    fn mv(&self, input: &[Item], output: &mut [Item]);
    //fn get_points(&self) -> Option<&[[f64; 3]]>;
    fn get_points(&self) -> Option<Vec<Point>>;
    fn rhs(&self) -> Vec<Self::Item>;
    fn get_condition_number(&self) -> Real<Self::Item>;
}

fn detect_python_env() -> (String, String) {
    let code = r#"
import sys
print(sys.executable)
print(sys.prefix)
"#;

    let output = std::process::Command::new("python3")
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

macro_rules! implement_structured_operator {
    ($scalar:ty, $mv:expr, $rhs_fn:expr) => {
        impl StructuredOperatorImpl<$scalar> for StructuredOperatorInterface {
            type Item = $scalar;

            fn new(params: &StructuredOperatorParams) -> Self {
                let class_name = params.structured_operator_type.as_ref();
                let c_str = std::ffi::CString::new(class_name).unwrap();
                let precision_str = match params.precision {
                    Precision::Double => std::ffi::CString::new("double").unwrap(),
                    Precision::Single => std::ffi::CString::new("single").unwrap(),
                };
                //let precision_str = std::ffi::CString::new(params.precision).unwrap();

                let geometry = std::ffi::CString::new(match params.geometry_type {
                    GeometryType::SphereSurface => "sphere_surface",
                    GeometryType::CubeSurface => "cube_surface",
                    GeometryType::CylinderSurface => "cylinder_surface",
                    GeometryType::EllipsoidSurface => "ellipsoid_surface",
                    GeometryType::TrefoilKnot => "trefoil_knot",
                    GeometryType::Sphere => "sphere",
                    GeometryType::Cube => "cube",
                    GeometryType::Satellite1 => "satellite1",
                })
                .unwrap();

                let raw = unsafe {
                    let (python_executable, _python_home) = detect_python_env();

                    let python_exe_c = CString::new(python_executable).unwrap();

                    initialize_structured_operator(
                        python_exe_c.as_ptr(),
                        c_str.as_ptr(),
                        params.dim_arg as f64,
                        geometry.as_ptr(),
                        params.kappa,
                        precision_str.as_ptr(),
                    )
                };

                assert!(!raw.is_null(), "Failed to initialize structured_operator");
                let n_points = unsafe { get_n_points(raw as *mut StructuredOperatorOpaque) };
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

            fn rhs(&self) -> Vec<$scalar> {
                $rhs_fn(&self).unwrap()
            }

            fn get_points(&self) -> Option<Vec<Point>> {
                get_bempp_points(&self)
            }

            fn get_condition_number(&self) -> Real<Self::Item> {
                unsafe { get_condition_number(self.raw) }
            }
        }
    };
}

pub fn rhs_real(structured_operator: &StructuredOperatorInterface) -> Option<Vec<f64>> {
    let ptr = unsafe { structured_operator_get_real_rhs(structured_operator.raw) };
    if ptr.is_null() {
        return None;
    }

    let total_len = structured_operator.n_points;
    let slice = unsafe { std::slice::from_raw_parts(ptr, total_len) };

    Some(slice.to_vec())
}

pub fn rhs_complex(structured_operator: &StructuredOperatorInterface) -> Option<Vec<c64>> {
    let ptr = unsafe { structured_operator_get_complex_rhs(structured_operator.raw) };
    if ptr.is_null() {
        return None;
    }
    let total_len = structured_operator.n_points;
    let slice = unsafe { std::slice::from_raw_parts(ptr, total_len) };
    Some(slice.to_vec())
}

implement_structured_operator!(f64, mv_structured_operator_real, rhs_real);
implement_structured_operator!(c64, mv_structured_operator_complex, rhs_complex);

pub fn one_dim_to_bempp_points(raw_points: &[f64]) -> Vec<bempp_octree::Point> {
    let points: Vec<_> = raw_points
        .chunks_exact(3)
        .map(|chunk| {
            let el = [chunk[0], chunk[1], chunk[2]];
            let point = bempp_octree::Point::new(el, 000);
            point
        })
        .collect();

    points
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

        let points: Vec<_> = raw_points
            .iter()
            .map(|&el| {
                let point = bempp_octree::Point::new(el, 000);
                point
            })
            .collect();
        points
    })
}

impl Drop for StructuredOperatorInterface {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                finalize_structured_operator(self.raw);
            }
            self.raw = std::ptr::null_mut();
        }
    }
}

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
    fn get_rhs(&self) -> Element<ConcreteElementContainer<ArrayVectorSpaceElement<Item>>>;
}

impl<Item: RlstScalar, Op: StructuredOperatorImpl<Item> + Shape<2>> Attr<Op, Item>
    for StructuredOperator<Item, Op>
where
    Op: StructuredOperatorImpl<Item, Item = Item>,
{
    fn get_rhs(&self) -> Element<ConcreteElementContainer<ArrayVectorSpaceElement<Item>>> {
        let mut vec: Element<ConcreteElementContainer<ArrayVectorSpaceElement<Item>>> =
            zero_element(self.domain.clone());
        let raw_data = self.op.rhs();
        for (i, val) in vec.imp_mut().view_mut().data_mut().iter_mut().enumerate() {
            *val = raw_data[i];
        }
        vec
    }
}

impl<Item: RlstScalar, Op: StructuredOperatorImpl<Item> + Shape<2>> std::fmt::Debug
    for StructuredOperator<Item, Op>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.op.shape();
        write!(f, "StructuredOperator: [{}x{}]", shape[0], shape[1]).unwrap();
        Ok(())
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
            TransMode::ConjNoTrans => {
                panic!("TransMode::ConjNoTrans not supported for multiplication.")
            }
            TransMode::Trans => {
                self.op
                    .mv(x.imp().view().data(), y.imp_mut().view_mut().data_mut());
            }
            TransMode::ConjTrans => {
                panic!("TransMode::ConjTrans not supported for multiplication.")
            }
        }
    }
}
