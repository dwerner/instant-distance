//! This is a simple example of how to use the instant-distance crate to build an index from a file.
//! The file is expected to be in the format of the fasttext word vectors.
//!
//! This example was built to performance using such tools as valgrind, perf, and heaptrack.
//!
//! Get the required file:
//! wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    time::Instant,
};

use instant_distance::{Builder, Metric, Search};
use instant_distance_py::{EuclidMetric, PointStorage};
use structopt::StructOpt;

#[derive(StructOpt)]
struct Opt {
    path: PathBuf,
    #[structopt(short = "w", default_value = "100_000")]
    word_count: usize,
}

fn main() -> Result<(), anyhow::Error> {
    let opt = Opt::from_args();

    let (words, points) = load_points(&opt.path, opt.word_count)?;

    println!("{} points loaded, building hnsw...", points.len());

    let start = Instant::now();
    let map = Builder::default()
        .build::<Vec<_>, [f32], EuclidMetric, String, PointStorage>(points, words);

    println!("indexing took {:?}", start.elapsed());

    // query
    let query_start = Instant::now();
    for _ in 0..100 {
        let mut search = Search::default();
        let point = vec![0.0; 304];
        let closest_point = map.search(&point, &mut search).next().unwrap();
    }
    println!("query took {:?}", query_start.elapsed());

    Ok(())
}

fn load_points(path: &Path, count: usize) -> Result<(Vec<String>, Vec<Vec<f32>>), anyhow::Error> {
    let mut words = vec![];
    let mut points = vec![];
    let mut reader = BufReader::new(File::open(path)?);

    // skip first line
    let mut discarded_line = String::new();
    let _bytes_read = reader.read_line(&mut discarded_line)?;

    for _ in 0..count {
        let mut line = String::new();
        let _read_bytes = reader.read_line(&mut line)?;
        let mut parts = line.split(' ');
        let word = parts.next().unwrap();
        words.push(word.to_string());
        let rest = parts.flat_map(|s| s.trim().parse::<f32>().ok()).collect();
        points.push(rest);
    }

    Ok((words, points))
}
