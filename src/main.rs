use fdeb_rs::fdeb_utils;
use fdeb_rs::{Edge, Fdeb, Vertex2D};
use serde::Serialize;
use std::collections::HashMap;

type Float = f32;

#[derive(Serialize)]
struct FdebResult {
    edges: Vec<Vec<Vertex2D>>,
    nodes: Vec<Vertex2D>,
}

fn main() {
    use std::fs::File;
    use std::io::prelude::*;

    let (points, edges) = fdeb_utils::read_graphml("airlines.xml");
    let rescaled = fdeb_utils::rescale(fdeb_utils::abs(&points), 900.0, 50.0, 460.0, 50.0);
    let rescaled_clone = (&rescaled).clone();
    let fdeb = Fdeb::new(rescaled, edges);

    for _ in 0..20 {
        let start = std::time::Instant::now();
        let result = fdeb.calculate_fdeb();
        let elapsed = start.elapsed();
        println!("Time to compute: {:?}", elapsed);
    }
    let start = std::time::Instant::now();
    let result = fdeb.calculate_fdeb();
    let elapsed = start.elapsed();
    println!("Time to compute: {:?}", elapsed);

    let serialized = serde_json::to_string(&FdebResult {
        edges: result,
        nodes: rescaled_clone.clone(),
    })
    .unwrap();
    let mut file = File::create("result.json").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();
}
