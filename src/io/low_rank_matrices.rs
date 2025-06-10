use rayon::prelude::*;
use rlst::prelude::*;
//Function that creates a low rank matrix by calculating a structured_operator given a random point distribution on an unit sphere.

pub trait ExpStructuredOperator: RlstScalar {
    fn exp_real_structured_operator(dist: Self, npoints: usize, kappa: Self) -> Self;
    fn exp_complex_structured_operator(dist: Self, npoints: usize, kappa: Self) -> num::Complex<Self>;
    fn laplace_structured_operator(dist: Self, npoints: usize, kappa: Self) -> Self;
    fn helmholtz_structured_operator(dist: Self, npoints: usize, kappa: Self) -> num::Complex<Self>;
}

macro_rules! implement_exp_structured_operator {
    ($scalar:ty) => {
        impl ExpStructuredOperator for $scalar {
            fn exp_real_structured_operator(dist: Self, npoints: usize, _kappa: Self) -> Self {
                let d: Self = num::NumCast::from(dist).unwrap();
                let n: Self = num::NumCast::from(npoints).unwrap();
                (1.0 / (n * (d * d).exp()))
            }
            fn exp_complex_structured_operator(dist: Self, npoints: usize, _kappa: Self) -> num::Complex<Self> {
                let d: Self = num::NumCast::from(dist).unwrap();
                let n: Self = num::NumCast::from(npoints).unwrap();
                let i = num::Complex::<Self>::new(0.0, 1.0);
                (1.0 / (n * (i * d * d).exp()))
            }
            fn laplace_structured_operator(dist: Self, npoints: usize, _kappa: Self) -> Self {
                let pi: Self = if std::any::TypeId::of::<Self>() == std::any::TypeId::of::<f32>() {
                    std::f32::consts::PI as Self
                } else {
                    std::f64::consts::PI as Self
                };
                let d: Self = num::NumCast::from(dist).unwrap();
                let n: Self = num::NumCast::from(npoints).unwrap();
                (1.0 / (4.0 * pi * n * d))
            }
            fn helmholtz_structured_operator(dist: Self, npoints: usize, kappa: Self) -> num::Complex<Self> {
                let pi: Self = if std::any::TypeId::of::<Self>() == std::any::TypeId::of::<f32>() {
                    std::f32::consts::PI as Self
                } else {
                    std::f64::consts::PI as Self
                };
                let d: Self = num::NumCast::from(dist).unwrap();
                let n: Self = num::NumCast::from(npoints).unwrap();
                let i = num::Complex::<Self>::new(0.0, 1.0);
                ((i * kappa * d).exp() / (4.0 * pi * n * d))
            }
        }
    };
}

implement_exp_structured_operator!(f32);
implement_exp_structured_operator!(f64);

pub trait LowRankMatrix: RlstScalar {
    fn get_real_matrix(
        points_x: &[bempp_octree::Point],
        structured_operator_fn: fn(dist: Self, npoints: usize, kappa: Self) -> Self,
        kappa: Self,
    ) -> DynamicArray<Self, 2>;
    fn get_complex_matrix(
        points_x: &[bempp_octree::Point],
        structured_operator_fn: fn(
            dist: Self,
            npoints: usize,
            kappa: <Self as RlstScalar>::Real,
        ) -> num::Complex<Self>,
        kappa: <Self as RlstScalar>::Real,
    ) -> DynamicArray<num::Complex<Self>, 2>
    where
        Self: PartialOrd;
}

macro_rules! implement_low_rank_matrix {
    ($scalar:ty) => {
        impl LowRankMatrix for $scalar {
            fn get_real_matrix(
                points_x: &[bempp_octree::Point],
                structured_operator_fn: fn(dist: Self, npoints: usize, kappa: Self) -> Self,
                kappa: Self,
            ) -> DynamicArray<Self, 2> {
                let start = std::time::Instant::now();
                let n = points_x.len();
                let num_chunks = rayon::current_num_threads();
                let chunk_size = (n + num_chunks - 1) / num_chunks;

                println!("Number of threads: {}", num_chunks);
                let widths_and_cols: Vec<_> = (0..n)
                    .step_by(chunk_size)
                    .map(|start| {
                        let end = (start + chunk_size).min(n);
                        (end - start, (start..end).collect::<Vec<_>>())
                    })
                    .collect();

                let mut col_chunks: Vec<_> = widths_and_cols
                    .par_iter()
                    .map(|(width, cols_inds)| {
                        let cols = rlst_dynamic_array2!(Self, [n, *width]);
                        let coords_vec_y = cols_inds
                            .iter()
                            .map(|col_ind| {
                                let coords_y = points_x[*col_ind].coords();
                                coords_y
                            })
                            .collect::<Vec<_>>();
                        (cols, coords_vec_y, cols_inds)
                    })
                    .collect();

                println!("Filling the columns");
                // Fill the columns in parallel
                // Use par_iter_mut to allow parallel iteration and mutation
                // Use enumerate to get the chunk number
                // Use for_each to iterate over the chunks
                col_chunks.par_iter_mut().enumerate().for_each(
                    |(chunk_num, (cols, coords_vec_y, _cols_inds))| {
                        println!("Started chunk {} with shape: {:?}", chunk_num, cols.shape());
                        let mut cols_view = cols.r_mut();
                        for (i, point_x) in points_x.iter().enumerate() {
                            let coords_x = point_x.coords();
                            for (j, coords_y) in coords_vec_y.iter().enumerate() {
                                let dist: <$scalar as RlstScalar>::Real = num::NumCast::from(
                                    ((coords_x[0] - coords_y[0]).powi(2)
                                        + (coords_x[1] - coords_y[1]).powi(2)
                                        + (coords_x[2] - coords_y[2]).powi(2))
                                    .sqrt(),
                                )
                                .unwrap();

                                let value = if dist > 0.0 {
                                    structured_operator_fn(dist, n, kappa)
                                } else {
                                    1.0.into()
                                };
                                cols_view[[i, j]] = value;
                            }
                        }
                        println!("Finished chunk {}", chunk_num);
                    },
                );

                let arr = rlst_dynamic_array2!(Self, [n, n]);
                let arr_mutex = std::sync::Mutex::new(arr);
                println!("Filling the matrix");
                col_chunks.into_par_iter().enumerate().for_each(
                    |(chunk_num, (cols, _coords_y, cols_inds))| {
                        println!("Filling chunk {}", chunk_num);
                        let mut arr_guard = arr_mutex.lock().unwrap();
                        let arr_view = arr_guard.r_mut();
                        arr_view
                            .into_subview([0, cols_inds[0]], [n, cols.shape()[1]])
                            .fill_from(cols.r());
                    },
                );

                let elapsed = start.elapsed();
                println!("Elapsed time: {:?}", elapsed);
                arr_mutex.into_inner().unwrap()
            }

            fn get_complex_matrix(
                points_x: &[bempp_octree::Point],
                structured_operator_fn: fn(
                    dist: Self,
                    npoints: usize,
                    kappa: <Self as RlstScalar>::Real,
                ) -> num::Complex<Self>,
                kappa: <Self as RlstScalar>::Real,
            ) -> DynamicArray<num::Complex<Self>, 2> {
                let n = points_x.len();
                let mut arr: DynamicArray<num::Complex<Self>, 2> =
                    rlst_dynamic_array2!(num::Complex<Self>, [n, n]);

                let mut cols: Vec<_> = (0..n)
                    .into_par_iter()
                    .map(|j| {
                        let point_y = points_x[j];
                        let coords_y = point_y.coords();
                        let col = rlst_dynamic_array2!(num::Complex<Self>, [n, 1]);
                        (col, coords_y)
                    })
                    .collect();

                cols.par_iter_mut().for_each(|(col, coords_y)| {
                    let mut col_view = col.r_mut();
                    for (i, point_x) in points_x.iter().enumerate() {
                        let coords_x = point_x.coords();
                        let dist: <$scalar as RlstScalar>::Real = num::NumCast::from(
                            ((coords_x[0] - coords_y[0]).powi(2)
                                + (coords_x[1] - coords_y[1]).powi(2)
                                + (coords_x[2] - coords_y[2]).powi(2))
                            .sqrt(),
                        )
                        .unwrap();

                        let value = if dist > 0.0 {
                            structured_operator_fn(dist, n, kappa)
                        } else {
                            1.0.into()
                        };
                        col_view[[i, 0]] = value;
                    }
                });

                cols.into_iter()
                    .enumerate()
                    .for_each(|(col_ind, (col, _coords_y))| {
                        let arr_view = arr.r_mut();
                        arr_view
                            .into_subview([0, col_ind], [n, 1])
                            .fill_from(col.r());
                    });

                arr
            }
        }
    };
}

implement_low_rank_matrix!(f32);
implement_low_rank_matrix!(f64);

pub trait StructuredOperatorMatrix: RlstScalar {
    fn get_exp_real_structured_operator_matrix(
        points_x: &[bempp_octree::Point],
        kappa: Self,
    ) -> DynamicArray<Self, 2>;
    fn get_exp_complex_structured_operator_matrix(
        points_x: &[bempp_octree::Point],
        kappa: Self,
    ) -> DynamicArray<num::Complex<Self>, 2>
    where
        Self: PartialOrd;
    fn get_laplace_matrix(points_x: &[bempp_octree::Point], kappa: Self) -> DynamicArray<Self, 2>;
    fn get_helmholtz_matrix(
        points_x: &[bempp_octree::Point],
        kappa: Self,
    ) -> DynamicArray<num::Complex<Self>, 2>
    where
        Self: PartialOrd;
}

macro_rules! implement_structured_operator_matrix {
    ($scalar:ty) => {
        impl StructuredOperatorMatrix for $scalar {
            fn get_exp_real_structured_operator_matrix(
                points_x: &[bempp_octree::Point],
                _kappa: Self,
            ) -> DynamicArray<Self, 2> {
                LowRankMatrix::get_real_matrix(points_x, Self::exp_real_structured_operator, 0.0)
            }

            fn get_exp_complex_structured_operator_matrix(
                points_x: &[bempp_octree::Point],
                kappa: Self,
            ) -> DynamicArray<num::Complex<Self>, 2> {
                LowRankMatrix::get_complex_matrix(points_x, Self::exp_complex_structured_operator, kappa)
            }

            fn get_laplace_matrix(
                points_x: &[bempp_octree::Point],
                _kappa: Self,
            ) -> DynamicArray<Self, 2> {
                LowRankMatrix::get_real_matrix(points_x, Self::laplace_structured_operator, 0.0)
            }

            fn get_helmholtz_matrix(
                points_x: &[bempp_octree::Point],
                kappa: Self,
            ) -> DynamicArray<num::Complex<Self>, 2> {
                LowRankMatrix::get_complex_matrix(points_x, Self::helmholtz_structured_operator, kappa)
            }
        }
    };
}

implement_structured_operator_matrix!(f32);
implement_structured_operator_matrix!(f64);
