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
    pub fn new(env: &mut Environment, inputs: usize, outputs: usize, depth: usize, width: usize, kernelsize: usize) -> Self {
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
        let x = input.apply(&self.conv1, ctx).leaky_relu(0.01);
        let x = x.apply(&self.conv2, ctx).leaky_relu(0.01);
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
        let x = x.apply(&self.conv3, ctx).leaky_relu(0.01);
        let x = x.apply(&self.conv4, ctx).leaky_relu(0.01);
        x
    }
}