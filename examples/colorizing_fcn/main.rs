#![feature(int_roundings)]
use descent::{module::*, prelude::*, optimizer::*, module::ModuleExt};
use image::{Rgb32FImage, DynamicImage, GenericImageView};
use rand::{thread_rng, RngCore, Rng};
use std::{io::Write, collections::HashMap};

mod fcn;
use crate::fcn::FCN;

fn main() {
    let mut rng = thread_rng();
    let mut env = Environment::new();

    let fcn = FCN::new(&mut env, 1, 3, 2, 16, 3);
    
    let image = image::open("images/Capybara_128px_square.jpg").expect("Could not open image");



    let image_bytes_bw = image.to_luma32f().to_vec();

    let input_param = env.static_parameter_with_data(
        [1, 128, 128, 1],
         "input", 
         &image_bytes_bw
    );

    //Compute output shape
    let output_shape = {
        let scope = env.scope();
        let x = scope.parameter(&input_param);
        let y = fcn.test(x);
        env.scope().build_graph();
        y.shape()
    };

    let output_shape_vec = output_shape.to_vec();

    eprintln!("Output shape {:?}", output_shape_vec);

    let border_x = (image.width() - output_shape_vec[2] as u32) / 2;
    let border_y = (image.height() - output_shape_vec[1] as u32) / 2;
    
    let cropped = image
    .view(border_x, border_y, image.width() - (2* border_x), image.height() - (2*border_y)).to_image();
    
    let image_bytes_rgb = DynamicImage::from(cropped)
    .to_rgb8()
    .to_vec()
    .into_iter()
    .map(|v| v as f32 / 255.0)
    .collect::<Vec<_>>();

    let target_param = env.static_parameter_with_data(
        output_shape, "target", &image_bytes_rgb);

    let output_param = env.static_parameter(output_shape, "output");

    let loss_param = env.static_parameter([1], "loss");

    //Create training graph
    let (train_graph, parameters, optimizer) = {
        let scope = env.scope();
        let x = fcn.train(scope.parameter(&input_param));
        //this is just mse loss
        let loss = (x - &target_param).square()
            .reduce_sum(-1, false)
            .reduce_sum(-1, false)
            .reduce_sum(-1, false)
            .set_loss(); 
        scope.update_parameter_value(&loss_param, |loss_sum| {
            loss_sum + loss.reduce_sum(0, false)
        });
        let parameters = scope.trainable_parameters();
        //Default optimizer
        let optimizer = Adam::new(
            &mut env,
            &scope,
            &parameters,
            0.04,
            0.99,
            0.999,
            1.0E-8
        );
        (scope.build_graph(), parameters, optimizer)
    };

    for param in parameters.iter(){
        env.reset_parameter(param, &mut rng);
    }

    let inference_graph = env.build_graph(|scope| {
        let input = scope.parameter(&input_param);
        let output = fcn.test(input);
        scope.write_parameter_value(&output_param, output.value());
    });

    let mut epoch = 1;
    eprintln!("Starting training");
    for _e in 0 .. 240{
        //Train for 10 steps
        for batch in 0 .. 10{
            env.writer(&loss_param).zero_fill();
            env.run(&train_graph, rng.next_u32());
            let loss = env.read_parameter_scalar(&loss_param) / output_shape_vec.iter().product::<usize>() as f32;
            eprint!("\rEpoch {epoch} Batch {batch} Loss={}      ", loss);
        }
        //Evaluate
        env.run(&inference_graph, rng.next_u32());

        let output_values = env.read_parameter_to_vec(&output_param);
        let scaled_image = Rgb32FImage::from_vec(output_shape[2] as u32, output_shape[1] as u32, output_values).unwrap();
        let scaled_image_u8 = DynamicImage::from(scaled_image).to_rgb8();
        eprintln!();
        scaled_image_u8.save_with_format(format!("capybara_colorized_fcn_epoch_{epoch:03}.jpg"), image::ImageFormat::Jpeg).unwrap();
        epoch += 1;
    }

}