#![feature(int_roundings)]
use descent::{module::*, prelude::*, optimizer::*, module::ModuleExt};
use rand::{thread_rng, RngCore, Rng};
use std::{io::Write, collections::HashMap};

//Unet definition, recursively holds all the conv layers
pub struct FCN {
    convs: Vec<Conv2D>
}
impl FCN {
    //Builder method
    pub fn new(env: &mut Environment, inputs: usize, outputs: usize, depth: usize, width: usize, kernelsize: usize) -> Self {
        let mut convs = Vec::new();
        convs.push(Conv2D::builder(inputs, width, kernelsize, kernelsize).build(env));

        for d in 0 .. depth{
            convs.push(Conv2D::builder(width, width, kernelsize, kernelsize).build(env));
        }

        convs.push(Conv2D::builder(width, outputs, kernelsize, kernelsize).build(env));
        Self { convs }
    }
}
impl Module for FCN {
    fn eval<'s>(&self, input: DualArray<'s>, ctx: &EvalContext) -> DualArray<'s> {
        let mut x = input;
        for conv in self.convs.iter(){
            x = x.apply(conv, ctx).leaky_relu(0.01);
        }
        x
    }
}