use crate::test_prep::DimArg;
use bempp_octree::Point;
use rlst::prelude::*;
use rlst::RlstScalar;
use std::rc::Rc;

#[repr(C)]
struct KernelOpaque {
    _private: [u8; 0],
}

extern "C" {
    fn initialize_kernel(
        class_name: *const i8,
        arg1: libc::c_double,
        geometry_type: *const i8,
        kappa: libc::c_double,
    ) -> *mut KernelOpaque;
    fn mv_kernel_real(
        kernel: *mut KernelOpaque,
        input: *const f64,
        output: *mut f64,
        len: libc::c_int,
    ) -> libc::c_int;
    fn mv_kernel_complex(
        kernel: *mut KernelOpaque,
        input: *const num::Complex<f64>, // or *const libc::c_void
        output: *mut num::Complex<f64>,
        len: libc::c_int,
    ) -> libc::c_int;
    fn get_points(kernel: *mut KernelOpaque) -> *const f64;
    fn get_condition_number(kernel: *mut KernelOpaque) -> f64;
    fn get_n_points(kernel: *mut KernelOpaque) -> usize;
    fn finalize_kernel(kernel: *mut KernelOpaque);
}

pub struct Kernel {
    raw: *mut KernelOpaque,
    pub n_points: usize,
}

#[derive(Debug, Clone)]
pub enum KernelType {
    Laplace,
    Helmholtz,
    Exp,
    BemLaplace,
    BemHelmholtz,
}

#[derive(Debug, Clone)]
pub enum GeometryType {
    SphereSurface,
    CubeSurface,
    CylinderSurface,
    EllipsoidSurface,
    TrefoilKnot,
    Sphere,
    Cube,
}

pub struct KernelParams {
    kernel_type: KernelType,
    geometry_type: GeometryType,
    dim_arg: DimArg,
    kappa: f64,
}

impl KernelParams {
    pub fn new(
        kernel_type: KernelType,
        geometry_type: GeometryType,
        dim_arg: DimArg,
        kappa: f64,
    ) -> Self {
        Self {
            kernel_type,
            geometry_type,
            dim_arg,
            kappa,
        }
    }
}

type Real<T> = <T as rlst::RlstScalar>::Real;
pub trait KernelImpl<Item: RlstScalar> {
    type Item: RlstScalar;
    fn new(params: KernelParams) -> Self;
    fn mv(&self, input: &[Item], output: &mut [Item]);
    //fn get_points(&self) -> Option<&[[f64; 3]]>;
    fn get_points(&self) -> Option<Vec<Point>>;
    fn get_condition_number(&self) -> Real<Self::Item>;
}

macro_rules! implement_kernel {
    ($scalar:ty, $mv:expr) => {
        impl KernelImpl<$scalar> for Kernel {
            type Item = $scalar;

            fn new(params: KernelParams) -> Self {
                let class_name = match params.kernel_type {
                    KernelType::Laplace => "LaplaceKernel",
                    KernelType::Helmholtz => "HelmholtzKernel",
                    KernelType::Exp => "ExpKernel",
                    KernelType::BemLaplace => "BemLaplaceKernel",
                    KernelType::BemHelmholtz => "BemHelmholtzKernel",
                };

                let c_str = std::ffi::CString::new(class_name).unwrap();

                let geometry = std::ffi::CString::new(match params.geometry_type {
                    GeometryType::SphereSurface => "sphere_surface",
                    GeometryType::CubeSurface => "cube_surface",
                    GeometryType::CylinderSurface => "cylinder_surface",
                    GeometryType::EllipsoidSurface => "ellipsoid_surface",
                    GeometryType::TrefoilKnot => "trefoil_knot",
                    GeometryType::Sphere => "sphere",
                    GeometryType::Cube => "cube",
                })
                .unwrap();

                let dim_arg = match params.dim_arg {
                    DimArg::NumPoints(num_points) => num_points as f64,
                    DimArg::MeshWidth(h) => h as f64,
                };
                let raw = unsafe {
                    initialize_kernel(
                        c_str.as_ptr(),
                        dim_arg as f64,
                        geometry.as_ptr(),
                        params.kappa,
                    )
                };

                assert!(!raw.is_null(), "Failed to initialize kernel");
                let n_points = unsafe { get_n_points(raw as *mut KernelOpaque) };
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
                        "mv_kernel call failed"
                    );
                }
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

pub fn get_bempp_points(kernel: &Kernel) -> Option<Vec<Point>> {
    let ptr = unsafe { get_points(kernel.raw) };
    if ptr.is_null() {
        return None;
    }
    let total_len = kernel.n_points * 3;
    let slice = unsafe { std::slice::from_raw_parts(ptr, total_len) };

    Some({
        let raw_points = unsafe {
            std::slice::from_raw_parts(slice.as_ptr() as *const [f64; 3], kernel.n_points)
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

implement_kernel!(f64, mv_kernel_real);
implement_kernel!(c64, mv_kernel_complex);

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { finalize_kernel(self.raw) };
    }
}

impl Shape<2> for Kernel {
    fn shape(&self) -> [usize; 2] {
        [self.n_points, self.n_points]
    }
}

pub struct KernelOperator<'a, Item: RlstScalar, Op: KernelImpl<Item> + Shape<2>> {
    op: &'a Op,
    domain: Rc<ArrayVectorSpace<Item>>,
    range: Rc<ArrayVectorSpace<Item>>,
}

impl<Item: RlstScalar, Op: KernelImpl<Item> + Shape<2>> std::fmt::Debug
    for KernelOperator<'_, Item, Op>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.op.shape();
        write!(f, "KernelOperator: [{}x{}]", shape[0], shape[1]).unwrap();
        Ok(())
    }
}

pub struct LocalOp<'a, Op> {
    pub op: &'a Op,
}

impl<Item: RlstScalar, Op: KernelImpl<Item> + Shape<2>> OperatorBase
    for KernelOperator<'_, Item, Op>
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

pub trait LocalFrom<'a, Op, Item: RlstScalar>: Sized {
    fn from_local(op: &'a Op) -> Self;
}

impl<'a, Item: RlstScalar, Op: KernelImpl<Item> + Shape<2>> LocalFrom<'a, Op, Item>
    for KernelOperator<'a, Item, Op>
{
    fn from_local(op: &'a Op) -> Self {
        let shape = op.shape();
        let domain = ArrayVectorSpace::from_dimension(shape[1]);
        let range = ArrayVectorSpace::from_dimension(shape[0]);
        KernelOperator { op, domain, range }
    }
}

impl<Item: RlstScalar, Op: KernelImpl<Item> + Shape<2>> AsApply for KernelOperator<'_, Item, Op> {
    fn apply_extended<
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
    ) {
        self.op
            .mv(x.imp().view().data(), y.imp_mut().view_mut().data_mut());
    }

    fn apply_extended_transpose<
        //TODO: Implement
        ContainerIn: ElementContainer<E = <Self::Domain as LinearSpace>::E>,
        ContainerOut: ElementContainerMut<E = <Self::Range as LinearSpace>::E>,
    >(
        &self,
        _alpha: <Self::Range as LinearSpace>::F,
        x: Element<ContainerIn>,
        _beta: <Self::Range as LinearSpace>::F,
        mut y: Element<ContainerOut>,
    ) {
        self.op
            .mv(x.imp().view().data(), y.imp_mut().view_mut().data_mut());
    }
}
