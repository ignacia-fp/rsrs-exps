use bempp_rsrs::rsrs::sketch::SketchData;
use mpi::topology::SimpleCommunicator;
use rand_distr::{Distribution, Standard, StandardNormal};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use rlst::{dense::tools::RandScalar, prelude::*};
use std::time::Instant;

use crate::io::{
    geometries::{cube_surface, randomly_distributed, sphere_surface},
    low_rank_matrices::KernelMatrix,
};

pub fn add_samples_parallel<
    Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse,
>(
    box_data: &mut SketchData<Item>,
    extra_num_samples: usize,
    arr: &DynamicArray<Item, 2>,
    silent: bool,
    _seed: u64,
) -> u128
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let start: Instant = Instant::now();
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
    let test_shape: [usize; 2] = box_data.test.shape();

    box_data
        .test
        .resize_in_place([box_data.dim, test_shape[1] + extra_num_samples]);
    box_data
        .sketch
        .resize_in_place([box_data.dim, test_shape[1] + extra_num_samples]);

    let mut extra_test = box_data
        .test
        .r_mut()
        .into_subview([0, test_shape[1]], [box_data.dim, extra_num_samples]);
    let mut extra_sketch = box_data
        .sketch
        .r_mut()
        .into_subview([0, test_shape[1]], [box_data.dim, extra_num_samples]);

    extra_test.fill_from_standard_normal(&mut rng);

    let num_chunks = rayon::current_num_threads();
    let chunk_size = (extra_num_samples + num_chunks - 1) / num_chunks;

    let mut sub: Vec<_> = (0..num_chunks)
        .into_iter()
        .map(|chunk_num| {
            let end_offset = (chunk_size * chunk_num).min(extra_num_samples);
            let offset = [0, end_offset];
            let current_chunk_size = chunk_size.min(extra_num_samples - end_offset);
            let shape = [box_data.dim, current_chunk_size];
            let mut sub_test = empty_array();
            sub_test.fill_from_resize(extra_test.r().into_subview(offset, shape));
            let mut sub_sketch = empty_array();
            sub_sketch.fill_from_resize(extra_sketch.r().into_subview(offset, shape));
            (sub_test, sub_sketch, offset, shape)
        })
        .collect();

    sub.par_iter_mut()
        .for_each(|(sub_test, sub_sketch, _offset, _shape)| {
            if !box_data.trans {
                sub_sketch.r_mut().simple_mult_into(arr.r(), sub_test.r());
            } else {
                sub_sketch.r_mut().mult_into(
                    TransMode::Trans,
                    TransMode::NoTrans,
                    num::One::one(),
                    arr.r(),
                    sub_test.r(),
                    num::Zero::zero(),
                );
            }
        });

    for (sub_test, sub_sketch, offset, shape) in sub {
        extra_sketch
            .r_mut()
            .into_subview(offset, shape)
            .fill_from(sub_sketch.r());
        extra_test
            .r_mut()
            .into_subview(offset, shape)
            .fill_from(sub_test.r());
    }
    let duration = start.elapsed();
    box_data.num_samples = test_shape[1] + extra_num_samples;

    if !silent {
        println!("Testing in {} ms", duration.as_millis());
    }

    duration.as_millis()
}

pub fn add_samples<Item: RlstScalar + RandScalar + MatrixId + MatrixInverse + MatrixPseudoInverse>(
    box_data: &mut SketchData<Item>,
    extra_num_samples: usize,
    arr: &DynamicArray<Item, 2>,
    silent: bool,
    _seed: u64,
) -> u128
where
    StandardNormal: Distribution<Item::Real>,
    Standard: Distribution<Item::Real>,
{
    let start: Instant = Instant::now();
    let mut rng: rand::prelude::ThreadRng = rand::thread_rng(); // For testing: ChaCha8Rng::seed_from_u64(0);
    let test_shape: [usize; 2] = box_data.test.shape();
    box_data
        .test
        .resize_in_place([box_data.dim, test_shape[1] + extra_num_samples]);
    box_data
        .sketch
        .resize_in_place([box_data.dim, test_shape[1] + extra_num_samples]);
    let mut sub_test = box_data
        .test
        .r_mut()
        .into_subview([0, test_shape[1]], [box_data.dim, extra_num_samples]);
    let mut sub_sketch = box_data
        .sketch
        .r_mut()
        .into_subview([0, test_shape[1]], [box_data.dim, extra_num_samples]);
    sub_test.fill_from_standard_normal(&mut rng);

    if !box_data.trans {
        sub_sketch.r_mut().simple_mult_into(arr.r(), sub_test.r());
    } else {
        sub_sketch.r_mut().mult_into(
            TransMode::Trans,
            TransMode::NoTrans,
            num::One::one(),
            arr.r(),
            sub_test.r(),
            num::Zero::zero(),
        );
    }
    let duration = start.elapsed();

    box_data.num_samples = test_shape[1] + extra_num_samples;

    if !silent {
        println!("Testing in {} ms", duration.as_millis());
    }

    duration.as_millis()
}

pub trait SampleTestFramework: RlstScalar {
    fn test<Item: RlstScalar>(geometry: &str, kernel: &str, npoints: &[usize]);
}

macro_rules! implement_sample_test_framework {
    ($scalar:ty) => {
        impl SampleTestFramework for $scalar {
            fn test<Item: RlstScalar>(geometry: &str, kernel: &str, npoints: &[usize]) {
                let universe: mpi::environment::Universe = mpi::initialize().unwrap();
                let comm: SimpleCommunicator = universe.world();

                let kernel_fn: fn(
                    &[bempp_octree::Point],
                    <$scalar as RlstScalar>::Real,
                ) -> DynamicArray<$scalar, 2> = if kernel == "standard_real" {
                    KernelMatrix::get_exp_real_kernel_matrix
                } else {
                    KernelMatrix::get_laplace_matrix
                };

                let geometry_fn: fn(usize, &SimpleCommunicator) -> Vec<bempp_octree::Point> =
                    if geometry == "cube" {
                        cube_surface
                    } else if geometry == "sphere" {
                        sphere_surface
                    } else {
                        randomly_distributed
                    };

                for &n in npoints.into_iter() {
                    println!("Num points: {}", n);
                    let points: Vec<bempp_octree::Point> = geometry_fn(n, &comm);
                    let arr: DynamicArray<$scalar, 2> = kernel_fn(&points, num::Zero::zero());
                    let test: DynamicArray<$scalar, 2> = empty_array();
                    let sketch: DynamicArray<$scalar, 2> = empty_array();
                    let mut sketching_data: SketchData<$scalar> = SketchData {
                        sketch,
                        test,
                        dim: arr.shape()[0],
                        num_samples: 0,
                        trans: false,
                    };
                    let test: DynamicArray<$scalar, 2> = empty_array();
                    let sketch: DynamicArray<$scalar, 2> = empty_array();
                    let mut sketching_parallel_data: SketchData<$scalar> = SketchData {
                        sketch,
                        test,
                        dim: arr.shape()[0],
                        num_samples: 0,
                        trans: false,
                    };
                    let extra_num_samples = [1500, 2000];
                    let mut times_1 = Vec::new();
                    let mut times_2 = Vec::new();

                    for &extra_samples in extra_num_samples.iter() {
                        println!("Extra samples: {}", extra_samples);
                        times_1.push(add_samples::<f64>(
                            &mut sketching_data,
                            extra_samples,
                            &arr,
                            true,
                            0,
                        ));
                        times_2.push(add_samples_parallel::<f64>(
                            &mut sketching_parallel_data,
                            extra_samples,
                            &arr,
                            true,
                            0,
                        ));
                    }

                    println!("Adding samples times: {:?}", times_1);
                    println!("Adding parallel samples times: {:?}", times_2);
                }
            }
        }
    };
}

implement_sample_test_framework!(f64);
