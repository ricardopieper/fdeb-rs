use rayon::prelude::*;
use rayon::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Float = f32;

const EPSILON: f32 = std::f32::EPSILON;
const EPS: Float = 1e-6;
const P_INITIAL: usize = 1;
const K: Float = 0.1; // global bundling constant controlling edge stiffness
const COMPATIBILITY_THRESHOLD: Float = 0.6;
const S_INITIAL: Float = 0.1; // init. distance to move points
const P_RATE: usize = 2; // subdivision rate increase
const C: i32 = 8; // number of cycles to perform
const I_INITIAL: usize = 90; // init. number of iterations for cycle
const I_RATE: Float = 2.0 / 3.0; // rate at which iteration number decreases i.e. 2/3

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex2D {
    pub x: Float,
    pub y: Float,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub target: usize,
    pub source: usize,
}

pub struct Fdeb {
    pub vertices: Vec<Vertex2D>,
    pub edges: Vec<Edge>,
}

impl Fdeb {
    pub fn new(vertices: Vec<Vertex2D>, edges: Vec<Edge>) -> Fdeb {
        let filtered_edges = Fdeb::filter_self_loops(&vertices, edges);
        Fdeb {
            vertices,
            edges: filtered_edges,
        }
    }

    fn vector_dot_product(&self, p: &Vertex2D, q: &Vertex2D) -> Float {
        p.x * q.x + p.y * q.y
    }

    //
    fn edge_as_vector(&self, p: &Edge) -> Vertex2D {
        Vertex2D {
            x: self.vertices[p.target].x - self.vertices[p.source].x,
            y: self.vertices[p.target].y - self.vertices[p.source].y,
        }
    }

    //
    fn edge_length(&self, node: &Edge) -> Float {
        // handling nodes that are on the same location, so that K/edge_length != Inf
        if (self.vertices[node.source].x - self.vertices[node.target].x).abs() < EPS
            && (self.vertices[node.source].y - self.vertices[node.target].y).abs() < EPS
        {
            EPS
        } else {
            self.euclidean_distance(&self.vertices[node.source], &self.vertices[node.target])
        }
    }

    //
    fn edge_midpoint(&self, e: &Edge) -> Vertex2D {
        let middle_x = (self.vertices[e.source].x + self.vertices[e.target].x) / 2.0;
        let middle_y = (self.vertices[e.source].y + self.vertices[e.target].y) / 2.0;

        Vertex2D {
            x: middle_x,
            y: middle_y,
        }
    }

    fn euclidean_distance(&self, p: &Vertex2D, q: &Vertex2D) -> Float {
        ((p.x - q.x).powi(2) + (p.y - q.y).powi(2)).sqrt()
    }

    //
    fn _compute_divided_edge_length_approx(
        &self,
        subdivision_points_for_edge: &[Vertex2D],
    ) -> Float {
        self.euclidean_distance(
            subdivision_points_for_edge.first().unwrap(),
            subdivision_points_for_edge.last().unwrap(),
        )
    }

    //
    fn compute_divided_edge_length(&self, subdivision_points_for_edge: &[Vertex2D]) -> Float {
        let from_first = subdivision_points_for_edge.iter().skip(1);
        let zipped = from_first.zip(subdivision_points_for_edge.iter());
        zipped
            .map(|(edge1, edge2)| self.euclidean_distance(edge1, edge2))
            .sum()
    }

    //
    fn project_point_on_line(
        &self,
        p: &Vertex2D,
        q_source: &Vertex2D,
        q_target: &Vertex2D,
    ) -> Vertex2D {
        let l = (q_target.x - q_source.x).powi(2) + (q_target.y - q_source.y).powi(2);

        let r = ((q_source.y - p.y) * (q_source.y - q_target.y)
            - (q_source.x - p.x) * (q_target.x - q_source.x))
            / l;

        Vertex2D {
            x: (q_source.x + r * (q_target.x - q_source.x)),
            y: (q_source.y + r * (q_target.y - q_source.y)),
        }
    }

    //
    fn initialize_edge_subdivisions(&self) -> Vec<Vec<Vertex2D>> {
        let mut subdivision_points_for_edges = Vec::<Vec<Vertex2D>>::new();

        for _ in 0..self.edges.len() {
            let subdivisions = Vec::<Vertex2D>::new();
            subdivision_points_for_edges.push(subdivisions);
        }

        subdivision_points_for_edges
    }

    //
    fn initialize_compatibility_lists(&self) -> HashMap<usize, Vec<usize>> {
        let mut compatibility_list_for_edge = HashMap::new();
        for i in 0..self.edges.len() {
            compatibility_list_for_edge.insert(i, Vec::<usize>::with_capacity(0));
            //0 compatible edges.
        }
        compatibility_list_for_edge
    }

    //
    fn apply_spring_force(
        &self,
        subdivision_points_for_edge: &[Vertex2D],
        i: usize,
        k_p: Float,
    ) -> Vertex2D {
        if subdivision_points_for_edge.len() < 3 {
            Vertex2D { x: 0.0, y: 0.0 }
        } else {
            let prev = &subdivision_points_for_edge[i - 1];
            let next = &subdivision_points_for_edge[i + 1];
            let crnt = &subdivision_points_for_edge[i];
            let x = prev.x - crnt.x + next.x - crnt.x;
            let y = prev.y - crnt.y + next.y - crnt.y;

            Vertex2D {
                x: x * k_p,
                y: y * k_p,
            }
        }
    }

    //
    fn apply_electrostatic_force(
        &self,
        subdivision_points_for_edge: &[Vec<Vertex2D>],
        compatible_edges_list: &[usize],
        i: usize,
        e_idx: usize,
    ) -> Vertex2D {
        if e_idx > subdivision_points_for_edge.len() - 1 ||
            i > subdivision_points_for_edge[e_idx].len() - 1 {
            Vertex2D { x: 0.0, y: 0.0 }
        } else {
            let edge = &subdivision_points_for_edge[e_idx][i];

            let (x, y) = compatible_edges_list
                .iter()
                .map(|oe| {

                    if *oe > subdivision_points_for_edge.len() - 1 ||
                        i > subdivision_points_for_edge[*oe].len() - 1 {
                        (0.0, 0.0)
                    }
                    else {
                        let edge_oe = &subdivision_points_for_edge[*oe][i];
                        let force_x = edge_oe.x - edge.x;
                        let force_y = edge_oe.y - edge.y;

                        if (force_x.abs() > EPS) || (force_y.abs() > EPS) {
                            let len = self.euclidean_distance(edge_oe, edge);
                            let diff = 1.0 / len;
                            (force_x * diff, force_y * diff)
                        } else {
                            (0.0, 0.0)
                        }
                    }
                })
                .fold((0.0, 0.0), |(acc_x, acc_y), (x, y)| (acc_x + x, acc_y + y));

            Vertex2D { x, y }
        }
    }

    //
    fn compute_forces_on_point(
        &self,
        e_idx: usize,
        s: Float,
        i: usize,
        k_p: Float,
        subdivision_points_for_edges: &[Vec<Vertex2D>],
        edge_subdivisions: &[Vertex2D],
        compatible_edges_list: &[usize],
    ) -> Vertex2D {
        let spring_force = self.apply_spring_force(edge_subdivisions, i, k_p);
        let electrostatic_force = self.apply_electrostatic_force(
            subdivision_points_for_edges,
            compatible_edges_list,
            i,
            e_idx,
        );

        Vertex2D {
            x: s * (spring_force.x + electrostatic_force.x),
            y: s * (spring_force.y + electrostatic_force.y),
        }
    }

    //
    fn compute_forces_on_points_iterator(
        &self,
        e_idx: usize,
        p: usize,
        s: Float,
        subdivision_points_for_edges: &[Vec<Vertex2D>],
        compatible_edges_list: &[usize],
    ) -> Vec<Vertex2D> {
        let edge_subdivisions = &subdivision_points_for_edges[e_idx];
        let k_p = K / (self.edge_length(&self.edges[e_idx]) * (p as Float + 1.0));
        (1..=p)
            .map(move |i| {
                self.compute_forces_on_point(
                    e_idx,
                    s,
                    i,
                    k_p,
                    subdivision_points_for_edges,
                    edge_subdivisions,
                    compatible_edges_list,
                )
            })
            .collect()
    }

    //
    fn apply_resulting_forces_on_subdivision_points(
        &self,
        e_idx: usize,
        p: usize,
        s: Float,
        subdivision_points_for_edges: &[Vec<Vertex2D>],
        compatible_edges_list: &[usize],
    ) -> Vec<Vertex2D> {
        self.compute_forces_on_points_iterator(
            e_idx,
            p,
            s,
            subdivision_points_for_edges,
            compatible_edges_list,
        )
    }

    //
    fn update_edge_divisions(
        &self,
        p: usize,
        subdivision_points_for_edge: &mut Vec<Vec<Vertex2D>>,
    ) {
        if p == 1 {
            let edge_subdivs = subdivision_points_for_edge
                .par_iter_mut()
                .zip(self.edges.par_iter());

            edge_subdivs.for_each(|(subdivisions, edge)| {
                *subdivisions = vec![
                    self.vertices[edge.source].clone(),
                    self.edge_midpoint(edge),
                    self.vertices[edge.target].clone(),
                ]
            });
        } else {
            let edge_subdivs = subdivision_points_for_edge
                .par_iter_mut()
                .zip(self.edges.par_iter());

            edge_subdivs.for_each(|(subdivisions, edge)| {
                let divided_edge_length = self.compute_divided_edge_length(subdivisions);
                let segment_length = divided_edge_length / (p + 1) as Float;
                let mut current_segment_length = segment_length;

                let mut new_subdivision_points =
                    Vec::<Vertex2D>::with_capacity(subdivisions.len() * 2);

                new_subdivision_points.push(self.vertices[edge.source].clone());

                for i in 1..subdivisions.len() {
                    let subdivision = &subdivisions[i];
                    let prev_subdivision = &subdivisions[i - 1];

                    let mut old_segment_length =
                        self.euclidean_distance(subdivision, prev_subdivision);

                    while old_segment_length > current_segment_length {
                        let percent_position = current_segment_length / old_segment_length;
                        let mut new_subdivision_point_x = prev_subdivision.x;
                        let mut new_subdivision_point_y = prev_subdivision.y;

                        new_subdivision_point_x +=
                            percent_position * (subdivision.x - prev_subdivision.x);
                        new_subdivision_point_y +=
                            percent_position * (subdivision.y - prev_subdivision.y);

                        old_segment_length -= current_segment_length;
                        current_segment_length = segment_length;

                        new_subdivision_points.push(Vertex2D {
                            x: new_subdivision_point_x,
                            y: new_subdivision_point_y,
                        });
                    }

                    current_segment_length -= old_segment_length;
                }
                new_subdivision_points.push(self.vertices[edge.target].clone());

                *subdivisions = new_subdivision_points;
            })
        }
    }

    fn angle_compatibility(&self, p: &Edge, q: &Edge) -> Float {
        (self.vector_dot_product(&self.edge_as_vector(p), &self.edge_as_vector(q))
            / (self.edge_length(p) * self.edge_length(q)))
        .abs()
    }

    fn scale_compatibility(&self, p: &Edge, q: &Edge) -> Float {
        let lavg = (self.edge_length(p) + self.edge_length(q)) / 2.0;
        2.0 / (lavg / self.edge_length(p).min(self.edge_length(q))
            + self.edge_length(p).max(self.edge_length(q)) / lavg)
    }

    fn position_compatibility(&self, p: &Edge, q: &Edge) -> Float {
        let lavg = (self.edge_length(p) + self.edge_length(q)) / 2.0;
        let mid_p = Vertex2D {
            x: (self.vertices[p.source].x + self.vertices[p.target].x) / 2.0,
            y: (self.vertices[p.source].y + self.vertices[p.target].y) / 2.0,
        };
        let mid_q = Vertex2D {
            x: (self.vertices[q.source].x + self.vertices[q.target].x) / 2.0,
            y: (self.vertices[q.source].y + self.vertices[q.target].y) / 2.0,
        };

        lavg / (lavg + self.euclidean_distance(&mid_p, &mid_q))
    }

    fn edge_visibility(&self, p: &Edge, q: &Edge) -> Float {
        let i_0 = self.project_point_on_line(
            &self.vertices[q.source],
            &self.vertices[p.source],
            &self.vertices[p.target],
        );
        let i_1 = self.project_point_on_line(
            &self.vertices[q.target],
            &self.vertices[p.source],
            &self.vertices[p.target],
        ); //send actual edge points positions
        let mid_i = Vertex2D {
            x: (i_0.x + i_1.x) / 2.0,
            y: (i_0.y + i_1.y) / 2.0,
        };

        let mid_p = Vertex2D {
            x: (self.vertices[p.source].x + self.vertices[p.target].x) / 2.0,
            y: (self.vertices[p.source].y + self.vertices[p.target].y) / 2.0,
        };

        (0.0 as Float).max(
            1.0 as Float
                - 2.0 * self.euclidean_distance(&mid_p, &mid_i)
                    / self.euclidean_distance(&i_0, &i_1),
        )
    }

    fn visibility_compatibility(&self, p: &Edge, q: &Edge) -> Float {
        self.edge_visibility(p, q).min(self.edge_visibility(q, p))
    }

    fn compatibility_score(&self, p: &Edge, q: &Edge) -> Float {
        self.angle_compatibility(p, q)
            * self.scale_compatibility(p, q)
            * self.position_compatibility(p, q)
            * self.visibility_compatibility(p, q)
    }

    fn are_compatible(&self, p: &Edge, q: &Edge) -> bool {
        self.compatibility_score(p, q) >= COMPATIBILITY_THRESHOLD
    }

    fn compute_compatibility_lists(
        &self,
        compatibility_list_for_edge: &mut HashMap<usize, Vec<usize>>,
    ) {
        (0..self.edges.len() - 1)
            .into_par_iter()
            .flat_map(|e| {
                (e + 1..self.edges.len())
                    .into_par_iter()
                    .filter(move |oe| self.are_compatible(&self.edges[e], &self.edges[*oe]))
                    .map(move |oe| (e, oe))
            })
            .collect::<Vec<(usize, usize)>>()
            .iter()
            .for_each(|(e, oe)| {
                {
                    let vec_e = compatibility_list_for_edge.get_mut(&e).unwrap();
                    vec_e.push(*oe);
                }
                {
                    let vec_oe = compatibility_list_for_edge.get_mut(&oe).unwrap();
                    vec_oe.push(*e);
                }
            })
    }

    //
    fn filter_self_loops(vertices: &[Vertex2D], edges: Vec<Edge>) -> Vec<Edge> {
        fn float_equals(x: Float, y: Float) -> bool {
            (x - y).abs() > EPSILON
        }

        edges
            .into_iter()
            .filter(|e| {
                let target_x = vertices[e.target].x;
                let target_y = vertices[e.target].y;
                let source_x = vertices[e.target].x;
                let source_y = vertices[e.target].y;
                !float_equals(source_x, target_x) || !float_equals(source_y, target_y)
            })
            .collect()
    }
    #[inline(never)]
    fn do_cycles(
        &self,
        mut edge_subdivisions: Vec<Vec<Vertex2D>>,
        compatibility_lists: &HashMap<usize, Vec<usize>>,
    ) -> Vec<Vec<Vertex2D>> {
        let mut s = S_INITIAL;
        let mut i = I_INITIAL;
        let mut p = P_INITIAL;

        for _ in 0..C {
            for _ in 0..i {
                //   let _ = Measure::measure(&format!("Iteration {}", iteration));
                let forces: Vec<Vec<Vertex2D>> = (0..self.edges.len())
                    .into_par_iter()
                    .map(|edge| {
                        self.apply_resulting_forces_on_subdivision_points(
                            edge,
                            p,
                            s,
                            &edge_subdivisions,
                            &compatibility_lists[&edge],
                        )
                    })
                    .collect();

                for edge in 0..self.edges.len() {
                    for ii in 0..p {
                        edge_subdivisions[edge][ii + 1].x += forces[edge][ii].x;
                        edge_subdivisions[edge][ii + 1].y += forces[edge][ii].y;
                    }
                }
            }
            s /= 2.0;
            p *= P_RATE;
            i = (I_RATE as Float * i as Float) as usize;

            self.update_edge_divisions(p, &mut edge_subdivisions);
        }
        edge_subdivisions
    }

    pub fn calculate_fdeb(&self) -> Vec<Vec<Vertex2D>> {
        let mut edge_subdivisions: Vec<Vec<Vertex2D>> = self.initialize_edge_subdivisions();
        let mut compatibility_lists = self.initialize_compatibility_lists();
        self.update_edge_divisions(P_INITIAL, &mut edge_subdivisions);

        //  let start = std::time::Instant::now();
        self.compute_compatibility_lists(&mut compatibility_lists);
        //  let elapsed = start.elapsed();
        //println!("\n\nTime to make compatibility lists: {:?}", elapsed);
        //   let start = std::time::Instant::now();
        let result = self.do_cycles(edge_subdivisions, &compatibility_lists);
        // let elapsed = start.elapsed();
        // println!("Time to run cycles: {:?}", elapsed);
        result
    }
}
