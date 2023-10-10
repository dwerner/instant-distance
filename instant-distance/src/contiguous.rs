use std::marker::PhantomData;

use rand::{rngs::SmallRng, Rng, SeedableRng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    types::{LayerId, Meta, INVALID},
    Builder, Point, PointId,
};

pub trait PointDataSource: Sync {
    fn decompose(&self) -> Vec<f32>;
    fn stride() -> usize;
}

pub struct PointIter<'a> {
    values: &'a [f32],
    order: &'a [usize],
    stride: usize,
    index: usize,
}

impl<'a> Iterator for PointIter<'a> {
    type Item = PointRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.order.len() {
            let i = self.order[self.index] * self.stride;
            self.index += 1;
            Some(PointRef(&self.values[i..i + self.stride]))
        } else {
            None
        }
    }
}

pub trait Storage<P: PointDataSource> {
    fn iter(&self) -> PointIter;
    fn get(&self, index: usize) -> Option<PointRef>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

pub struct PointRef<'a>(pub &'a [f32]);

impl<'a> PointRef<'a> {
    pub fn from_data(values: &'a [f32]) -> Self {
        Self(values)
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Default)]
pub struct ContiguousStorage<T: PointDataSource> {
    pub values: Vec<f32>,
    pub order: Vec<usize>,
    _phantom: PhantomData<T>,
}

impl<P: PointDataSource> ContiguousStorage<P> {
    pub(crate) fn empty() -> Self {
        Self {
            values: Vec::new(),
            order: Vec::new(),
            _phantom: PhantomData,
        }
    }
    pub(crate) fn new(
        points: &[P],
        meta: &Meta,
        builder: Builder,
    ) -> (Self, Vec<(LayerId, PointId)>, Vec<PointId>) {
        let mut rng = SmallRng::seed_from_u64(builder.seed);
        assert!(points.len() < u32::MAX as usize);
        let mut shuffled = (0..points.len())
            .map(|i| (PointId(rng.gen_range(0..points.len() as u32)), i))
            .collect::<Vec<_>>();
        shuffled.sort_unstable();

        let mut order = Vec::with_capacity(points.len());
        let mut layer_assignments = Vec::with_capacity(points.len());
        let mut out = vec![INVALID; points.len()];
        let mut at_layer = meta.next_lower(None).unwrap();
        for (i, (_, idx)) in shuffled.into_iter().enumerate() {
            let pid = PointId(layer_assignments.len() as u32);
            if i == at_layer.1 {
                at_layer = meta.next_lower(Some(at_layer.0)).unwrap();
            }

            order.push(idx);
            layer_assignments.push((at_layer.0, pid));
            out[idx] = pid;
        }
        debug_assert_eq!(
            layer_assignments.first().unwrap().0,
            LayerId(meta.len() - 1)
        );
        debug_assert_eq!(layer_assignments.last().unwrap().0, LayerId(0));

        (
            Self {
                values: points
                    .iter()
                    .flat_map(PointDataSource::decompose)
                    .collect::<Vec<_>>(),
                order,
                _phantom: PhantomData,
            },
            layer_assignments,
            out,
        )
    }
}

impl<T: PointDataSource> Storage<T> for ContiguousStorage<T> {
    fn iter(&self) -> PointIter {
        PointIter {
            values: &self.values,
            order: &self.order,
            stride: T::stride(),
            index: 0,
        }
    }

    fn get(&self, index: usize) -> Option<PointRef> {
        self.order
            .get(index)
            .map(|&i| PointRef(&self.values[i * T::stride()..(i + 1) * T::stride()]))
    }

    fn len(&self) -> usize {
        self.order.len()
    }

    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

impl<'a> Point for PointRef<'a> {
    fn distance(&self, other: &Self) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::{
                _mm256_add_ps, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_loadu_ps,
                _mm256_mul_ps, _mm256_setzero_ps, _mm256_sub_ps, _mm_add_ps, _mm_add_ss,
                _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
            };
            debug_assert_eq!(self.0.len(), other.0.len());

            unsafe {
                let mut acc_8x = _mm256_setzero_ps();
                for (lh_slice, rh_slice) in self.0.chunks_exact(8).zip(other.0.chunks_exact(8)) {
                    let lh_8x = _mm256_loadu_ps(lh_slice.as_ptr());
                    let rh_8x = _mm256_loadu_ps(rh_slice.as_ptr());
                    let diff = _mm256_sub_ps(lh_8x, rh_8x);
                    let diff_squared = _mm256_mul_ps(diff, diff);
                    acc_8x = _mm256_add_ps(diff_squared, acc_8x);
                }

                // Sum up the components in `acc_8x`
                let acc_high = _mm256_extractf128_ps(acc_8x, 1);
                let acc_low = _mm256_castps256_ps128(acc_8x);
                let acc_4x = _mm_add_ps(acc_high, acc_low);

                let mut acc = _mm_add_ps(acc_4x, _mm_movehl_ps(acc_4x, acc_4x));
                acc = _mm_add_ss(acc, _mm_shuffle_ps(acc, acc, 0x55));

                let remaining_elements = &self.0[self.0.len() - self.0.len() % 8..];
                let mut residual = 0.0;
                for (&lh, &rh) in remaining_elements
                    .iter()
                    .zip(other.0[self.0.len() - other.0.len() % 8..].iter())
                {
                    residual += (lh - rh).powi(2);
                }

                let residual = residual + _mm_cvtss_f32(acc);
                residual.sqrt()
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MyPoint {
        values: Vec<f32>,
    }

    impl PointDataSource for MyPoint {
        fn decompose(&self) -> Vec<f32> {
            self.values.clone()
        }

        fn stride() -> usize {
            2
        }
    }

    fn create_contiguous_storage(
        points: Vec<MyPoint>,
        order: Vec<usize>,
    ) -> ContiguousStorage<MyPoint> {
        ContiguousStorage {
            values: points.iter().flat_map(MyPoint::decompose).collect(),
            order,
            _phantom: PhantomData,
        }
    }

    #[test]
    fn test_point_iter_stride() {
        let points = vec![
            MyPoint {
                values: vec![1.0, 2.0],
            },
            MyPoint {
                values: vec![3.0, 4.0],
            },
        ];
        let order = vec![1, 0];
        let storage = create_contiguous_storage(points, order);

        let expected_points = vec![vec![3.0, 4.0], vec![1.0, 2.0]];
        for (i, point_ref) in storage.iter().enumerate() {
            assert_eq!(point_ref.0, expected_points[i].as_slice());
        }
    }
}
