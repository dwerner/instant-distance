use std::marker::PhantomData;

use rand::{rngs::SmallRng, Rng, SeedableRng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{
    types::{LayerId, Meta, INVALID},
    Builder, Element, Point, PointId,
};

pub struct PointIter<'a, E, P: Point<Element = E>> {
    values: &'a [P::Element],
    order: &'a [usize],
    index: usize,
    _marker: PhantomData<&'a P>,
}

impl<'a, E, P: Point<Element = E>> Iterator for PointIter<'a, E, P> {
    type Item = PointRef<'a, E, P>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.order.len() {
            let i = self.order[self.index] * P::STRIDE;
            self.index += 1;
            Some(PointRef(&self.values[i..i + P::STRIDE]))
        } else {
            None
        }
    }
}

pub trait Storage<E, P: Point<Element = E>> {
    fn iter(&self) -> PointIter<'_, E, P>;
    fn get(&self, index: usize) -> Option<PointRef<'_, E, P>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
}

// TODO: remove default N
pub struct PointRef<'a, E, P: Point<Element = E>>(pub &'a [P::Element]);

impl<'a, E, P: Point<Element = E>> PointRef<'a, E, P> {
    pub fn from_data(values: &'a [P::Element]) -> Self {
        Self(values)
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Default)]
pub struct ContiguousStorage<E: Element, P: Point<Element = E>> {
    pub values: Vec<E>,
    pub order: Vec<usize>,
    _phantom: PhantomData<P>,
}

impl<E: Element, P: Point<Element = E>> ContiguousStorage<E, P> {
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
                    .flat_map(Point::as_slice)
                    .copied()
                    .collect::<Vec<_>>(),
                order,
                _phantom: PhantomData,
            },
            layer_assignments,
            out,
        )
    }
}

impl<'a, E: Element + 'a, P: Point<Element = E>> Storage<E, P> for ContiguousStorage<E, P> {
    fn iter(&self) -> PointIter<'_, E, P> {
        PointIter {
            values: &self.values,
            order: &self.order,
            index: 0,
            _marker: PhantomData,
        }
    }

    fn get(&self, index: usize) -> Option<PointRef<'_, E, P>> {
        self.order
            .get(index)
            .map(|&i| PointRef(&self.values[i * P::STRIDE..(i + 1) * P::STRIDE]))
    }

    fn len(&self) -> usize {
        self.order.len()
    }

    fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

// blanket impl
impl<E: Element, P: Point<Element = E>> Point for PointRef<'_, E, P> {
    const STRIDE: usize = P::STRIDE;
    type Element = E;

    fn as_slice(&self) -> &[Self::Element] {
        self.0
    }

    fn distance(&self, other: &Self) -> f32 {
        Element::distance(self.0, other.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::simd::distance_simd_f32;

    use super::*;

    struct MyPoint {
        values: Vec<f32>,
    }

    impl Point for MyPoint {
        type Element = f32;
        fn as_slice(&self) -> &[f32] {
            &self.values
        }

        fn distance(&self, other: &Self) -> f32 {
            let zelf = PointRef::<f32, MyPoint>::from_data(self.as_slice());
            let other = PointRef::<f32, MyPoint>::from_data(other.as_slice());
            distance_simd_f32(&zelf.0, &other.0)
        }

        const STRIDE: usize = 2;
    }

    fn create_contiguous_storage(
        points: Vec<MyPoint>,
        order: Vec<usize>,
    ) -> ContiguousStorage<f32, MyPoint> {
        ContiguousStorage {
            values: points
                .iter()
                .flat_map(MyPoint::as_slice)
                .map(|i| *i)
                .collect(),
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
