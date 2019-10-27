use criterion::{criterion_group, criterion_main, Criterion, ParameterizedBenchmark};

use fdeb_rs::*;
use fdeb_rs::fdeb_utils;

fn criterion_benchmark(c: &mut Criterion) {

    c.bench("calculate_fdeb", ParameterizedBenchmark::new("fdeb", 
            |b, _| { 
                let (points, edges) = fdeb_utils::read_graphml("airlines.xml");
                let rescaled = fdeb_utils::rescale(fdeb_utils::abs(&points), 900.0, 50.0, 460.0, 50.0);
                let fdeb = Fdeb::new(rescaled, edges);
                b.iter(|| fdeb.calculate_fdeb())
            }, 0..1)
        .sample_size(15));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);