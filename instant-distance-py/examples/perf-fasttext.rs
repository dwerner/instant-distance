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
    thread::sleep,
    time::{Duration, Instant},
};

use instant_distance::{contiguous, Builder, Search};
use instant_distance_py::FloatArray;
use rand::{rngs::ThreadRng, Rng};
use structopt::StructOpt;

#[global_allocator]
static ALLOC: tracy_full::alloc::GlobalAllocator = tracy_full::alloc::GlobalAllocator::new();

#[derive(StructOpt)]
struct Opt {
    #[structopt(
        short = "f",
        parse(from_os_str),
        default_value = "../instant-distance-benchmarking/wiki.en.align.vec"
    )]
    path: PathBuf,
    #[structopt(short = "w", default_value = "50000")]
    word_count: usize,

    #[structopt(short = "n", default_value = "1000")]
    num_queries: usize,

    #[structopt(short = "c")]
    contiguous: bool,

    #[structopt(short = "s")]
    wait: bool,
}

fn main() -> Result<(), anyhow::Error> {
    let opt = Opt::from_args();
    let (words, points) = load_points(&opt.path, opt.word_count)?;
    println!("{} points loaded, building hnsw...", points.len());

    let seed = ThreadRng::default().gen();

    if opt.contiguous {
        contiguous(opt.wait, seed, opt.num_queries, points.clone())?;
    } else {
        original(opt.wait, seed, opt.num_queries, words, points)?;
    }

    Ok(())
}

fn original(
    wait: bool,
    seed: u64,
    num_queries: usize,
    words: Vec<String>,
    points: Vec<FloatArray>,
) -> Result<(), anyhow::Error> {
    let bar = indicatif::ProgressBar::new(points.len() as u64);
    let start = Instant::now();
    let map = Builder::default()
        .seed(seed)
        .progress(bar)
        .build::<FloatArray, String>(points, words);
    println!("original indexing took {:?}", start.elapsed());

    if wait {
        println!("sleeping for 15s");
        sleep(Duration::from_millis(15000));
    }

    let mut search = Search::default();
    let point = FloatArray([0.0; 300]);
    for _ in 0..20 {
        let query_start = Instant::now();
        for _ in 0..num_queries {
            let _closest_point = map.search(&point, &mut search).next().unwrap();
            tracy_full::frame!("original search");
        }
        tracy_full::frame!("search group");
        println!("{} queries took {:?}", num_queries, query_start.elapsed());
    }
    tracy_full::frame!();
    Ok(())
}

fn contiguous(
    wait: bool,
    seed: u64,
    num_queries: usize,
    points: Vec<FloatArray>,
) -> Result<(), anyhow::Error> {
    let bar = indicatif::ProgressBar::new(points.len() as u64);
    let start = Instant::now();
    let (hnsw, _ids) = Builder::default()
        .seed(seed)
        .progress(bar)
        .build_contiguous::<FloatArray, _>(points, vec![0]);
    println!("contiguous indexing took {:?}", start.elapsed());

    if wait {
        println!("sleeping for 15s");
        sleep(Duration::from_millis(15000));
    }

    let mut search = contiguous::Search::default();
    let point = FloatArray([0.0; 300]);
    for _ in 0..20 {
        let query_start = Instant::now();
        for _ in 0..num_queries {
            let _closest_point = hnsw.search(&point, &mut search).next().unwrap();
            tracy_full::frame!("contiguous search");
        }

        tracy_full::frame!("search group");
        println!("{} queries took {:?}", num_queries, query_start.elapsed());
    }
    tracy_full::frame!();
    Ok(())
}

fn load_points(path: &Path, count: usize) -> Result<(Vec<String>, Vec<FloatArray>), anyhow::Error> {
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
        let rest = parts
            .flat_map(|s| s.trim().parse::<f32>().ok())
            .collect::<Vec<_>>();

        let mut float_array_inner = [0f32; 300];
        float_array_inner.copy_from_slice(&rest[..300]);
        let float_array = FloatArray(float_array_inner);

        points.push(float_array);
    }

    Ok((words, points))
}
