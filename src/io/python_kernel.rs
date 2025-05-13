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
        kappa: libc::c_double,
    ) -> *mut KernelOpaque;
    fn mv_kernel(
        kernel: *mut KernelOpaque,
        input: *const f64,
        output: *mut f64,
        len: libc::c_int,
    ) -> libc::c_int;
    fn get_points(kernel: *mut KernelOpaque) -> *const f64;
    fn get_n_points(kernel: *mut KernelOpaque) -> usize;
    fn finalize_kernel(kernel: *mut KernelOpaque);
}

pub struct Kernel {
    raw: *mut KernelOpaque,
    pub n_points: usize,
}

type Real<T> = <T as rlst::RlstScalar>::Real;

pub enum KernelType{
    Laplace,
    Helmholtz,
    Exp,
    BemLaplace,
}

pub struct KernelParams<Item: RlstScalar>{
    kernel_type: KernelType,
    n_points: usize,
    kappa: Real<Item>,
    h: Real<Item>
}

impl <Item: RlstScalar> KernelParams<Item>{
    pub fn new(kernel_type: KernelType,
    n_points: usize,
    kappa: Real<Item>,
    h: Real<Item>) -> Self{
        Self{kernel_type, n_points, kappa, h}
    }
}

pub trait KernelAttr<Item: RlstScalar> {
    type Item: RlstScalar;
    fn new(params: KernelParams<Item>) -> Self;
    fn mv(&self, input: &[Item], output: &mut [Item]);
    fn get_points(&self) -> Option<&[[f64; 3]]>;
    fn get_bempp_points(&self) -> Vec<Point>;
}


impl KernelAttr<f64> for Kernel {
    type Item = f64;

    fn new(params: KernelParams<Self::Item>) -> Self {

        let class_name = match params.kernel_type{
            KernelType::Laplace => "LaplaceKernel",
            KernelType::Helmholtz => "HelmholtzKernel",
            KernelType::Exp => "ExpKernel",
            KernelType::BemLaplace => "BemLaplaceKernel",
        };


        let c_str = std::ffi::CString::new(class_name).unwrap();

        let raw = if matches!(params.kernel_type, KernelType::BemLaplace){
            unsafe { initialize_kernel(c_str.as_ptr(), params.h as f64, params.kappa) }
        }
        else{
             unsafe { initialize_kernel(c_str.as_ptr(), params.n_points as f64, params.kappa) }
        };
        
        assert!(!raw.is_null(), "Failed to initialize kernel");
        let n_points = unsafe { get_n_points(raw as *mut Kernel) };
        Self { raw, n_points }
    }

    fn mv(&self, input: &[Self::Item], output: &mut [Self::Item]) {
        assert_eq!(input.len(), self.n_points);
        assert_eq!(output.len(), self.n_points);
        unsafe {
            assert!(
                mv_kernel(
                    self.raw,
                    input.as_ptr(),
                    output.as_mut_ptr(),
                    self.n_points as libc::c_int
                ) != 0,
                "mv_kernel call failed"
            );
        }

    }

    fn get_points(&self) -> Option<&[[f64; 3]]> {
        let ptr = unsafe { get_points(self.raw) };
        if ptr.is_null() {
            return None;
        }
        let total_len = self.n_points * 3;
        let slice = unsafe { std::slice::from_raw_parts(ptr, total_len) };
        Some(unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const [f64; 3], self.n_points) })
    }

    fn get_bempp_points(&self) -> Vec<Point> {
        let raw_points = self.get_points().unwrap();
        let points = raw_points
            .iter()
            .map(|&el| {
                let point = bempp_octree::Point::new(el, 000);
                point
            })
            .collect();

        points
    }
}

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

pub struct KernelOperator<'a, Item: RlstScalar, Op: KernelAttr<Item> + Shape<2>> {
    op: &'a Op,
    domain: Rc<ArrayVectorSpace<Item>>,
    range: Rc<ArrayVectorSpace<Item>>,
}

impl<Item: RlstScalar, Op: KernelAttr<Item> + Shape<2>> std::fmt::Debug
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

impl<Item: RlstScalar, Op: KernelAttr<Item> + Shape<2>> OperatorBase
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

impl<'a, Item: RlstScalar, Op: KernelAttr<Item> + Shape<2>> LocalFrom<'a, Op, Item>
    for Operator<KernelOperator<'a, Item, Op>>
{
    fn from_local(op: &'a Op) -> Self {
        let shape = op.shape();
        let domain = ArrayVectorSpace::from_dimension(shape[1]);
        let range = ArrayVectorSpace::from_dimension(shape[0]);
        Self::new(KernelOperator { op, domain, range })
    }
}

impl<Item: RlstScalar, Op: KernelAttr<Item> + Shape<2>> AsApply for KernelOperator<'_, Item, Op> {
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
}
