use std::cmp::max;
use std::ops::{Deref, Index, Range};

use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::contiguous::{Hnsw, M};
use crate::types::{Layer, LayerId, NearestIter};
use crate::PointId;

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Debug, Default)]
pub(crate) struct Meta(pub(crate) Vec<LayerMeta>);

impl Meta {
    pub(crate) fn new(ml: f32, mut num: usize) -> Self {
        let mut inner = Vec::new();
        let mut neighbors = 0;
        loop {
            let mut next = (num as f32 * ml) as usize;
            if next < M {
                next = 0;
            }

            let start = neighbors;
            neighbors += num * M * if inner.len() == 0 { 2 } else { 1 };
            inner.push(LayerMeta {
                max: num - next,
                total: num,
                start,
                end: neighbors,
            });

            if next == 0 {
                break;
            }
            num = next;
        }

        Self(inner)
    }

    pub(crate) fn next_lower(&self, cur: Option<LayerId>) -> Option<(LayerId, usize)> {
        let idx = cur.map(|l| l.0 - 1).unwrap_or(self.len() - 1);
        self.0.get(idx).map(|meta| (LayerId(idx), meta.total))
    }

    pub(crate) fn layer<'a>(&self, layer: LayerId, neighbors: &'a [PointId]) -> LayerSlice<'a> {
        let meta = &self.0[layer.0];
        LayerSlice {
            neighbors: &neighbors[meta.start..meta.end],
            stride: if layer.is_zero() { M * 2 } else { M },
        }
    }

    pub(crate) fn layers_mut<'a>(
        &self,
        mut neighbors: &'a mut [PointId],
    ) -> Vec<LayerSliceMut<'a>> {
        let mut layers = Vec::with_capacity(self.0.len());
        let mut pos = 0;
        for meta in self.0.iter() {
            let len = meta.end - meta.start;
            let stride = if pos == 0 { M * 2 } else { M };
            let (cur, rest) = neighbors.split_at_mut(len);
            layers.push(LayerSliceMut {
                neighbors: cur,
                stride,
            });

            neighbors = rest;
            pos += len;
        }

        layers
    }

    pub(crate) fn descending<'a>(&'a self) -> impl Iterator<Item = LayerId> + 'a {
        (0..self.0.len()).into_iter().rev().map(LayerId)
    }

    pub(crate) fn points(&self, layer: LayerId) -> Range<usize> {
        let meta = &self.0[layer.0];
        max(meta.total - meta.max, 1)..meta.total
    }

    pub(crate) fn neighbors(&self) -> usize {
        self.0.last().unwrap().end
    }

    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }
}

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Debug)]
pub(crate) struct LayerMeta {
    max: usize,
    total: usize,
    start: usize,
    end: usize,
}

pub(crate) struct LayerSliceMut<'a> {
    neighbors: &'a mut [PointId],
    stride: usize,
}

impl<'a> LayerSliceMut<'a> {
    pub(crate) fn copy_from_zero(&mut self, zero: &[RwLock<ZeroNode<'_>>]) {
        let stride = self.stride;
        self.neighbors
            .par_chunks_mut(stride)
            .zip(zero)
            .for_each(|(dst, src)| {
                dst.copy_from_slice(&src.read()[..stride]);
            });
    }

    pub(crate) fn zero_nodes(&mut self) -> Vec<RwLock<ZeroNode<'_>>> {
        self.neighbors
            .chunks_exact_mut(self.stride)
            .map(|n| RwLock::new(ZeroNode(n)))
            .collect::<Vec<_>>()
    }

    pub(crate) fn as_ref(&self) -> LayerSlice<'_> {
        LayerSlice {
            neighbors: self.neighbors.as_ref(),
            stride: self.stride,
        }
    }
}

pub(crate) struct LayerSlice<'a> {
    neighbors: &'a [PointId],
    stride: usize,
}

impl<'a> Layer for LayerSlice<'a> {
    type Slice = &'a [PointId];

    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice> {
        let start = pid.0 as usize * self.stride;
        let end = start + self.stride;
        assert!(self.neighbors.len() >= end);
        NearestIter::new(&self.neighbors[start..end])
    }
}

#[derive(Debug)]
pub(crate) struct ZeroNode<'a>(pub(crate) &'a mut [PointId]);

impl<'a> ZeroNode<'a> {
    pub(crate) fn rewrite(&mut self, mut iter: impl Iterator<Item = PointId>) {
        for slot in self.0.iter_mut() {
            if let Some(pid) = iter.next() {
                *slot = pid;
            } else if *slot != INVALID {
                *slot = INVALID;
            } else {
                break;
            }
        }
    }

    pub(crate) fn insert(&mut self, idx: usize, pid: PointId) {
        // It might be possible for all the neighbor's current neighbors to be closer to our
        // neighbor than to the new node, in which case we skip insertion of our new node's ID.
        if idx >= self.0.len() {
            return;
        }

        if self.0[idx].is_valid() {
            let end = (M * 2) - 1;
            self.0.copy_within(idx..end, idx + 1);
        }

        self.0[idx] = pid;
    }

    pub(crate) fn set(&mut self, idx: usize, pid: PointId) {
        self.0[idx] = pid;
    }
}

impl<'a> Deref for ZeroNode<'a> {
    type Target = [PointId];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> Layer for &'a [RwLock<ZeroNode<'a>>] {
    type Slice = MappedRwLockReadGuard<'a, [PointId]>;

    fn nearest_iter(&self, pid: PointId) -> NearestIter<Self::Slice> {
        NearestIter::new(RwLockReadGuard::map(
            self[pid.0 as usize].read(),
            Deref::deref,
        ))
    }
}

impl<P> Index<PointId> for Hnsw<P> {
    type Output = P;

    fn index(&self, index: PointId) -> &Self::Output {
        &self.points[index.0 as usize]
    }
}

impl<'a> Index<PointId> for [RwLock<ZeroNode<'a>>] {
    type Output = RwLock<ZeroNode<'a>>;

    fn index(&self, index: PointId) -> &Self::Output {
        &self[index.0 as usize]
    }
}

pub(crate) const INVALID: PointId = PointId(u32::MAX);
