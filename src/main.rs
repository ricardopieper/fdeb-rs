use fdeb_rs::fdeb_utils;
use fdeb_rs::fdeb_svg;
use fdeb_rs::{Edge, Fdeb, Vertex2D};
use serde::Serialize;
use std::collections::HashMap;

type Float = f32;

#[derive(Serialize)]
struct FdebResult {
    edges: Vec<Vec<Vertex2D>>,
    nodes: Vec<Vertex2D>,
}
use rand::distributions::{Distribution, Uniform};
fn sample(edges: Vec<Edge>, percentage: f32) -> Vec<Edge> {

    let mut rng = rand::thread_rng();
    let die = Uniform::from(0.0 .. 1.0);

    let mut result = vec![];

    for edge in edges {

        let throw = die.sample(&mut rng);

        if throw < percentage {
            result.push(edge);
        }

    }
    
    result
}

fn main() {
    use std::fs::File;
    use std::io::prelude::*;

    let (points, edges) = fdeb_utils::read_json("br-migration.json");

    let edges = sample(edges, 0.03);

    println!("Loaded {} points and {} edges", points.len(), edges.len());

    let rescaled = fdeb_utils::rescale(fdeb_utils::abs_translate(points),
        1050.0, 50.0, 
        50.0, 1050.0);

    let rescaled_clone = (&rescaled).clone();
    let fdeb = Fdeb::new(rescaled, edges);

    let start = std::time::Instant::now();
    let result = fdeb.calculate_fdeb();
    let elapsed = start.elapsed();
    println!("Time to compute: {:?}", elapsed);
    
    
    /*let serialized = serde_json::to_string(&FdebResult {
        edges: result,
        nodes: rescaled_clone.clone(),
    })
    .unwrap();
    let mut file = File::create("result.json").unwrap();
    file.write_all(serialized.as_bytes()).unwrap();*/

    fdeb_svg::render(&result, &rescaled_clone, "assets/template-brazil.svg",  "assets/style.css");

}
