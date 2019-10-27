
use crate::fdeb::*;
use std::fs::File;
use std::io::prelude::*;
use serde::Deserialize;
pub fn read_graphml(file: &str) -> (Vec<Vertex2D>, Vec<Edge>) {
    use roxmltree::*;

    let mut xml_str = String::new();
    File::open(file)
        .unwrap()
        .read_to_string(&mut xml_str)
        .unwrap();
    let parsed = Document::parse(&xml_str).unwrap();
    let root: roxmltree::Node = parsed.root_element();
    let graph = root
        .children()
        .find(|child| child.tag_name().name() == "graph")
        .unwrap();

    let points: Vec<Vertex2D> = graph
        .children()
        .filter(|node: &roxmltree::Node| node.tag_name().name() == "node")
        .map(|node: roxmltree::Node| {
            let x: roxmltree::Node = node
                .children()
                .find(|n| match n.attribute("key") {
                    Some(x) => x == "x",
                    None => false,
                })
                .unwrap();

            let y: roxmltree::Node = node
                .children()
                .find(|n| match n.attribute("key") {
                    Some(x) => x == "y",
                    None => false,
                })
                .unwrap();

            Vertex2D {
                x: x.text().unwrap().parse().unwrap(),
                y: y.text().unwrap().parse().unwrap(),
            }
        })
        .collect();

    let edges: Vec<crate::fdeb::Edge> = graph
        .children()
        .filter(|node: &roxmltree::Node| node.tag_name().name() == "edge")
        .map(|node: roxmltree::Node| {
            let source: usize = node.attribute("source").unwrap().parse().unwrap();
            let target: usize = node.attribute("target").unwrap().parse().unwrap();
            crate::fdeb::Edge { source, target }
        })
        .collect();

    (points, edges)
}

pub fn read_json(file: &str) -> (Vec<Vertex2D>, Vec<Edge>) {
    
    #[derive(Deserialize)]
    struct Format {
        nodes: Vec<Vertex2D>,
        edges: Vec<Edge>
    }

    let mut json_str = String::new();
    File::open(file)
        .unwrap()
        .read_to_string(&mut json_str)
        .unwrap();

    let deserialized: Format = serde_json::from_str(&json_str).unwrap();
    
    (deserialized.nodes, deserialized.edges)
}

pub fn domain_transform(
    value: Float,
    source_min: Float,
    source_max: Float,
    target_min: Float,
    target_max: Float,
) -> Float {
    let source_range = source_max - source_min;
    let target_range = target_max - target_min;

    let offset_in_source = value - source_min;

    target_min + ((offset_in_source / source_range) * target_range)
}

pub fn rescale(
    raw_points: Vec<Vertex2D>,
    x_range_start: Float,
    x_range_end: Float,
    y_range_start: Float,
    y_range_end: Float,
) -> Vec<Vertex2D> {
    let mut all_x: Vec<Float> = raw_points.iter().map(|x| x.x).collect();
    all_x.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut all_y: Vec<Float> = raw_points.iter().map(|x| x.y).collect();
    all_y.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_x = all_x[0];
    let min_y = all_y[0];
    let max_x = *all_x.last().unwrap();
    let max_y = *all_y.last().unwrap();

    raw_points
        .iter()
        .map(|x: &Vertex2D| Vertex2D {
            x: domain_transform(x.x, min_x, max_x, x_range_start, x_range_end),
            y: domain_transform(x.y, min_y, max_y, y_range_start, y_range_end),
        })
        .collect()
}


pub fn abs_translate(raw_points: Vec<Vertex2D>) -> Vec<Vertex2D> {
    let mut all_x: Vec<Float> = raw_points.iter().map(|x| x.x).collect();
    all_x.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut all_y: Vec<Float> = raw_points.iter().map(|x| x.y).collect();
    all_y.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_x = all_x[0];
    let min_y = all_y[0];

    raw_points
        .iter()
        .map(|x: &Vertex2D| Vertex2D {
            x: (x.x + min_x).abs(),
            y: (x.y + min_y).abs(),
        })
        .collect()
}

pub fn abs(raw_points: &[Vertex2D]) -> Vec<Vertex2D> {
    raw_points
        .iter()
        .map(|x: &Vertex2D| Vertex2D {
            x: x.x.abs(),
            y: x.y.abs(),
        })
        .collect()
}