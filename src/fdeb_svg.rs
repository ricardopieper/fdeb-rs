use crate::fdeb::*;
use std::fs::File;
use std::io::prelude::*;
pub fn render(
    lines: &[Vec<Vertex2D>],
    cities: &[Vertex2D],
    template_file: &str,
    style_file: &str) {
   
   let mut svg_template = String::new();
   File::open(template_file)
        .unwrap()
        .read_to_string(&mut svg_template)
        .unwrap();

   let mut style = String::new();
   File::open(style_file)
        .unwrap()
        .read_to_string(&mut style)
        .unwrap();

   svg_template = svg_template.replace("@import url(style.css);", &style);

   let line_connections = lines.iter().map(|line| {
      let line_str = line.iter().map(|point| {
         format!("{},{}", point.x, point.y)
      }).collect::<Vec<_>>().join("L");
      format!("<path d=\"M{}\" />\n", line_str)
   }).collect::<Vec<_>>().join("");

   //<circle r="0.5" cx="204.1662" cy="432.45306"/>
   let cities = cities.iter().map(|city| {
         format!("<circle r=\"0.5\" cx=\"{}\" cy=\"{}\" />\n", city.x, city.y)
   }).collect::<Vec<_>>().join("");

   svg_template = svg_template.replace("<import-lines />", &line_connections);
   svg_template = svg_template.replace("<import-cities />", &cities);
   
    let mut file = File::create("output-map.svg").unwrap();
    file.write_all(svg_template.as_bytes()).unwrap();
   /*
   use resvg::prelude::*;
   let opt = resvg::Options::default();
   let rtree = usvg::Tree::from_str(&svg_template, &opt.usvg).unwrap();
   let mut output = resvg::default_backend().render_to_image(&rtree, &opt).unwrap();
   output.save_png(std::path::Path::new("output-map.svg"));*/
}