#![feature(int_roundings)]
use descent::{module::*, prelude::*, optimizer::*, module::ModuleExt};
use image::{Rgb32FImage, DynamicImage};
use rand::{thread_rng, RngCore, Rng};
use std::{io::Write, collections::HashMap};

fn main() {
    let mut rng = thread_rng();
    let mut env = Environment::new();
    
    let image = image::open("images/Capybara_128px_square.jpg").expect("Could not open image");

    let image_bytes_rgb = image
        .to_rgb8()
        .to_vec()
        .into_iter()
        .map(|v| v as f32 / 255.0)
        .collect::<Vec<_>>();

    let input_param = env.static_parameter_with_data(
        [1, 128, 128, 3],
         "input", 
         &image_bytes_rgb
    );

    let output_param = env.static_parameter([1,64,64,3], "output");

    let graph = env.build_graph(|scope| {
        let input = scope.parameter(&input_param);
        let output = input.crop(32, 32, 32, 32);
        scope.write_parameter_value(&output_param, output.value());
    });

    env.run(&graph, rng.next_u32());

    let output_values = env.read_parameter_to_vec(&output_param);

    let scaled_image = Rgb32FImage::from_vec(image.width() / 2, image.height() / 2, output_values).unwrap();

    let scaled_image_u8 = DynamicImage::from(scaled_image).to_rgb8();

    scaled_image_u8.save_with_format("capybara_cropped.jpg", image::ImageFormat::Jpeg).unwrap();

}