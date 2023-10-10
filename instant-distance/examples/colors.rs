use instant_distance::{Builder, PointDataSource, PointRef, Search};

fn main() {
    let points = vec![Point(255, 0, 0), Point(0, 255, 0), Point(0, 0, 255)];
    let values = vec!["red", "green", "blue"];

    let map = Builder::default().build(&points, values);
    let mut search = Search::default();

    let burnt_orange = Point(204, 85, 0);
    let data = burnt_orange.decompose();
    let point_ref = PointRef::from_data(&data);

    let closest_point = map.search(&point_ref, &mut search).next().unwrap();

    println!("{:?}", closest_point.value);
}

#[derive(Clone, Copy, Debug)]
struct Point(isize, isize, isize);

impl instant_distance::PointDataSource for Point {
    fn decompose(&self) -> Vec<f32> {
        vec![self.0 as f32, self.1 as f32, self.2 as f32]
    }

    fn stride() -> usize {
        3
    }
}
