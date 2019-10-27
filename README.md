fdeb.rs
=======

Port from a JS implementation of the FDEB (Force-directed edge bundling) algorithm in this repository https://github.com/upphiminn/d3.ForceBundle.

For the airlines dataset, the original algorithm takes a couple seconds to calculate in my machine, while this Rust port takes around 180ms. Using this port
I processed a graph with 46k edges and 5500 vertices.

This is a work in progress, so I haven't created a crate, neither have I properly documented it.