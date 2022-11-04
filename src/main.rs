#![feature(int_roundings)]
use descent::{module::*, prelude::*, optimizer::*, module::ModuleExt};
use rand::{thread_rng, RngCore, Rng};
use std::{io::Write, collections::HashMap};

//Unet definition, recursively holds all the conv layers
pub struct UNet {
    conv1: Conv2D,
    conv2: Conv2D,
    inner: 
        Option<(
            MaxPool2D,
            Box<Self>
        )>,
    conv3: Conv2D,
    conv4: Conv2D
}
impl UNet {
    //Builder method
    fn new(env: &mut Environment, inputs: usize, outputs: usize, depth: usize, width: usize, kernelsize: usize) -> Self {
        Self {
            conv1: Conv2D::builder(inputs, width, kernelsize, kernelsize).build(env),
            conv2: Conv2D::builder(width, width, kernelsize, kernelsize).build(env),
            inner: if depth > 0 {Some((
                MaxPool2D::default(),
                Box::new(Self::new(env, width, width * 2, depth - 1, width * 2, kernelsize))
            ))} else {None},
            conv3: Conv2D::builder(if depth > 0 {width * 3} else {width}, width, kernelsize, kernelsize).build(env),
            conv4: Conv2D::builder(width, outputs, kernelsize, kernelsize).build(env)
        }
    }
}
impl Module for UNet {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        let x = input.apply(&self.conv1, ctx);
        let x = x.apply(&self.conv2, ctx);
        let x = if let Some((pool, inner)) = self.inner.as_ref() {
            let [_, h_outer, w_outer, _]: [usize; 4] = x.shape().try_into().unwrap();

            let x_inner = x.apply(pool, ctx);
            let x_inner = inner.eval(x_inner, ctx);
            let [_, h_inner, w_inner, _]: [usize; 4] = x_inner.shape().try_into().unwrap();
            let x_inner = x_inner.upsample(
                w_outer.div_ceil(w_inner),             
                h_outer.div_ceil(h_inner)
            );
            let [_, h_inner, w_inner, _]: [usize; 4] = x_inner.shape().try_into().unwrap();
            assert_eq!(h_inner, w_inner);
            let left = (w_inner - w_outer) / 2;
            let right = (w_inner - w_outer) - left;
            let top = (h_inner - h_outer) / 2;
            let bottom = (h_inner - h_outer) - top;
            let x_inner = x_inner.crop(
                left, top, right, bottom
            );
            x.concat(x_inner, -1)
        } else {
            x
        };
        let x = x.apply(&self.conv3, ctx);
        let x = x.apply(&self.conv4, ctx);
        x
    }
}




fn main() {
    let mut rng = thread_rng();
    let mut env = Environment::new();
    let unet = UNet::new(&mut env, 1,1,0,8,3);

    //Define batch size
    let batch_size = 8;

    //Define params
    let x_param = env.static_parameter([batch_size, 64, 64, 1], "input");
    let loss_param = env.static_parameter([1], "loss");
        
    //Compute output shape
    let output_shape = {
        let scope = env.scope();
        let x = scope.parameter(&x_param);
        let y = unet.test(x);
        env.scope().build_graph();
        y.shape()
    };

    //Output parameter with the correct shape
    let y_param = env.static_parameter(output_shape, "output");

    //Create training graph
    let (train_graph, parameters, optimizer) = {
        let scope = env.scope();
        let x = unet.train(scope.parameter(&x_param));
        //this is just mse loss
        let loss = (x - &y_param).square()
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
            0.001,
            0.9,
            0.99,
            1.0E-8
        );
        (scope.build_graph(), parameters, optimizer)
    };
    eprintln!("Training graph created");

    //Reset parameters and loss
    for param in parameters.iter(){
        env.reset_parameter(param, &mut rng);
    }

    //Create some data
    let mut xs = Vec::new();
    let mut ys = Vec::new();

    let xlen = x_param.shape().iter().product::<usize>();
    let ylen = y_param.shape().iter().product::<usize>();

    for _ in 0 ..batch_size{
        let x_this_batch = (0 .. xlen / batch_size).into_iter().map(|i| i as f32 / xlen as f32).collect::<Vec<_>>();
        let y_this_batch = (0 .. ylen / batch_size).into_iter().map(|i| 1.0 - (i as f32 / ylen as f32)).collect::<Vec<_>>();
        xs.extend(x_this_batch);
        ys.extend(y_this_batch);
    }

    //Upload data
    for i in 0 .. 100{
        //Reset loss
        env.writer(&loss_param).zero_fill();
        //eprintln!("X wants {} values, gets {}", x_param.shape().iter().product::<usize>(), xs.len());
        let mut x_writer = env.writer(&x_param);
        x_writer.write_all(bytemuck::cast_slice(&xs)).unwrap();
        drop(x_writer);
        //eprintln!("Y wants {} values, gets {}", y_param.shape().iter().product::<usize>(), ys.len());
        let mut y_writer = env.writer(&y_param);
        y_writer.write_all(bytemuck::cast_slice(&ys)).unwrap();
        drop(y_writer);

        //Run graph
        env.run(&train_graph, rng.next_u32());
        //Get and print loss
        let train_loss = env.read_parameter_scalar(&loss_param) / batch_size as f32;
        eprintln!("Epoch {} Training loss: {}", i, train_loss);
    }

    let mut serialized_parameters = HashMap::new();
    for (index, param) in parameters.iter().enumerate(){
        if param.is_trainable(){
            let value = env.read_parameter_to_vec(param);
            let name = param.name();
            eprintln!("Index {} Name {} => Shape {:?}, Value ({} floats)", index, param.name(), param.shape(), value.len());
            serialized_parameters.insert((index, name), (param.shape(), value));
        }
    }

    //Create execution graph
    //See https://git.geomar.de/valentin-buck/geofeaturesegmentation/-/blob/59a20d3ae5fbb6382c8e269db503849251694bd8/geofeaturesegmentation/src/network.rs for actually doing this
    let execution_graph= env.build_graph(|scope| {
        let x = unet.test(scope.parameter(&x_param));
        scope.write_parameter_value(&y_param, x.value())
    });
    eprintln!("Execution graph created");

    eprintln!("Execution before serialization");
    let mut x_writer = env.writer(&x_param);
        x_writer.write_all(bytemuck::cast_slice(&xs)).unwrap();
        drop(x_writer);
    env.run(&execution_graph, rng.next_u32());
    let output = env.read_parameter_to_vec(&y_param);

    let unet2 = UNet::new(&mut env, 1,1,0,8,3);
    let (unet2_execution_graph, unet2_parameters) = {
        let scope = env.scope();
        let x = unet2.test(scope.parameter(&x_param));
        scope.write_parameter_value(&y_param, x.value());
        let parameters = scope.trainable_parameters();
        (scope.build_graph(), parameters)
    };

    for (index, param) in unet2_parameters.iter().enumerate(){
        let name = param.name();
        let key = (index, name);
        if let Some((shape, value)) = serialized_parameters.get(&key){
            assert_eq!(shape.to_vec(), param.shape().to_vec());
            let mut writer = env.writer(param);
            writer.write_all(bytemuck::cast_slice(value)).unwrap();
            eprintln!("Uploaded {:?}", key);
        }else{
            eprintln!("Parameter not found for key {:?}", key);
        }
    }

    eprintln!("Execution after serialization");
    let mut x_writer = env.writer(&x_param);
        x_writer.write_all(bytemuck::cast_slice(&xs)).unwrap();
        drop(x_writer);
    env.run(&unet2_execution_graph, rng.next_u32());
    let unet2_output = env.read_parameter_to_vec(&y_param);

    assert_eq!(output, unet2_output);

}
