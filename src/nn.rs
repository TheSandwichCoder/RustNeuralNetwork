use rand::Rng;
use std::fs;
use crate::activation_functions::*;


fn activation(x: f32, is_activation: bool) -> f32{
    if !is_activation{
        return x;
    }

    return ReLu(x);
    // return sigmoid(x);
}

fn deriv_activation(x: f32, is_activation: bool) -> f32{
    if !is_activation{
        return 1.0;
    }

    return deriv_ReLu(x);
    // return deriv_sigmoid(x);
}

fn sequential_add(vec1: &mut Vec<f32>, vec2: &Vec<f32>, length: usize){
    for i in 0..length{
        vec1[i] += vec2[i];
    }
}

pub struct Perceptron{
    n_inputs: usize,
    
    weights: Vec<f32>,
    gradients: Vec<f32>,
    momentum: Vec<f32>,

    bias: f32,
    bias_gradient: f32, 
    bias_momentum: f32,
    is_activation: bool,
}

impl Perceptron{
    pub fn new(n_inputs: usize, is_activation: bool) -> Perceptron{
        let mut perc = Perceptron{n_inputs: n_inputs, weights: Vec::new(), gradients: Vec::new(), momentum: Vec::new(), bias: 0.0, bias_gradient: 0.0, bias_momentum: 0.0, is_activation: is_activation};
        let mut rng = rand::thread_rng();

        for i in 0..n_inputs{
            perc.weights.push(rng.gen_range(-1.0..1.0));
            perc.momentum.push(0.0);
            perc.gradients.push(0.0);
            // perc.weights.push(0.5);
        }

        perc.bias = rng.gen_range(-1.0..1.0);
        // perc.bias = 0.5;
        
        return perc;
    }

    pub fn forward(&self, inputs: &Vec<f32>) -> f32{
        let mut sum: f32 = 0.0;

        for input_i in 0..self.n_inputs{
            sum += inputs[input_i] * self.weights[input_i];
        }

        sum += self.bias;

        return activation(sum, self.is_activation);
    }

    pub fn reset_gradients(&mut self){
        for gradient_i in 0..self.n_inputs{
            self.gradients[gradient_i] = 0.0;
        }
        self.bias_gradient = 0.0;
    }

    pub fn update_weights(&mut self){
        for weight_i in 0..self.n_inputs{
            self.momentum[weight_i] = mr * self.momentum[weight_i] + self.gradients[weight_i] * lr;

            self.weights[weight_i] += self.momentum[weight_i];
        }

        self.bias_momentum = self.bias_momentum * mr + self.bias_gradient * lr;
        self.bias += self.bias_momentum
    }

    pub fn backward(&mut self, inputs: &Vec<f32>, z: f32, part_deriv: f32){
        let s = deriv_activation(z, self.is_activation);

        for input_i in 0..self.n_inputs{
            self.gradients[input_i] += inputs[input_i] * part_deriv * s;
        }
        self.bias_gradient += s * part_deriv;
    }

    pub fn get_part_deriv(&self, z: f32, part_deriv: f32) -> Vec<f32>{
        let s = deriv_activation(z, self.is_activation);

        let mut part_derivs: Vec<f32> = vec![0.0; self.n_inputs];

        for i in 0..self.n_inputs{
            part_derivs[i] = s * self.weights[i] * part_deriv;
        }

        return part_derivs;
    }
}

const lr: f32 = -0.0005; // 0.0003
const mr: f32 = 0.9;

pub struct NeuralNetwork{
    n_layers: usize,
    layer_dim: Vec<usize>,
    layer_inputs: Vec<Vec<f32>>,

    n_inputs: usize,
    n_outputs: usize,
    
    part_derivs: Vec<Vec<Vec<f32>>>,
    perc_layers: Vec<Vec<Perceptron>>,
}

impl NeuralNetwork{
    pub fn new(layers: Vec<usize>) -> NeuralNetwork{
        let mut nn = NeuralNetwork{n_layers: layers.len(), layer_dim: layers.clone(), layer_inputs: Vec::new(), n_inputs: 0, n_outputs: 0, part_derivs: Vec::new(), perc_layers: Vec::new()};
        
        // initialise the array of layer inputs
        for layer_i in 0..nn.n_layers{
            nn.layer_inputs.push(vec![0.0; nn.layer_dim[layer_i]]);
        }

        nn.n_inputs = nn.layer_dim[0];
        nn.n_outputs = nn.layer_dim[nn.n_layers - 1];
        
        for layer_i in 1..nn.n_layers{
            nn.part_derivs.push(Vec::new());
            nn.perc_layers.push(Vec::new());

            // initialise the array of partial derivatives
            for perc_j in 0..nn.layer_dim[layer_i]{
                nn.part_derivs[layer_i - 1].push(vec![0.0; nn.layer_dim[layer_i - 1]]);
            }

            let is_activation = layer_i != nn.n_layers - 1;

            // initialises the array of perceptrons
            for perc_i in 0..nn.layer_dim[layer_i]{
                nn.perc_layers[layer_i - 1].push(Perceptron::new(nn.layer_dim[layer_i - 1], is_activation))
            }
        }

        return nn;
    }

    pub fn get_outputs(&self) -> Vec<f32>{
        return (self.layer_inputs[self.n_layers - 1]).clone()
    }

    pub fn forward(&mut self, inputs: Vec<f32>){
        for input_i in 0..self.n_inputs{
            self.layer_inputs[0][input_i] = inputs[input_i];
        }

        for layer_i in 1..self.n_layers{
            // let mut prev_input_layer = &mut self.layer_inputs[layer_i - 1];

            let perc_layer = &self.perc_layers[layer_i - 1];

            for input_i in 0..self.layer_dim[layer_i]{
                // self.layer_inputs[layer_i][input_i] = 0;

                self.layer_inputs[layer_i][input_i] = perc_layer[input_i].forward(&mut self.layer_inputs[layer_i - 1]);
            }
        }
    }

    pub fn show_info(&self){
        for layer_i in 1..self.n_layers{
            println!("layer {}", layer_i);
            for perc_i in 0..self.layer_dim[layer_i]{
                println!("weights:{:?} bias:{}", self.perc_layers[layer_i - 1][perc_i].weights, self.perc_layers[layer_i - 1][perc_i].bias);
            }
        }
    }

    pub fn save(&self){
        let mut file_txt = String::new();

        // Layer settings
        let lyr_setting_txt = self.layer_dim
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        file_txt.push_str(&lyr_setting_txt);
        file_txt.push('\n');

        // Perceptron weights and biases
        for perc_layer in &self.perc_layers {
            for perc in perc_layer {
                let weights_str = perc.weights
                    .iter()
                    .map(|w| w.to_string())
                    .collect::<Vec<_>>()
                    .join(" ");
                file_txt.push_str(&weights_str);
                file_txt.push_str(&format!(" {}\n", perc.bias));
            }
        }

        fs::write("neuralnets/nn1.txt", file_txt);
    }

    pub fn backward(&mut self, errors: Vec<f32>){
        
        // resets all the gradients
        for layer_i in 1..self.n_layers{
            for perc_i in 0..self.layer_dim[layer_i]{
                self.perc_layers[layer_i - 1][perc_i].reset_gradients();
            }
        }

        // main back propogation loop
        for layer_i in (1..self.n_layers).rev(){
            let perc_layer = &mut self.perc_layers[layer_i -  1];
            let curr_layer_length = self.layer_dim[layer_i - 1];

            for perc_i in 0..self.layer_dim[layer_i]{
                let mut perc = &mut perc_layer[perc_i];

                let mut part_deriv = vec![0.0 ; curr_layer_length];
                let z = self.layer_inputs[layer_i][perc_i];

                if layer_i + 1 >= self.n_layers{
                    let perc_part_deriv = errors[perc_i];

                    part_deriv = perc.get_part_deriv(z, perc_part_deriv);
                    perc.backward(&self.layer_inputs[layer_i - 1], z,  perc_part_deriv);
                }

                else{
                    for der_i in 0..self.layer_dim[layer_i + 1]{
                        let perc_part_deriv = self.part_derivs[layer_i][der_i][perc_i];

                        // update the partial deriv
                        let semi_part_deriv = perc.get_part_deriv(z, perc_part_deriv);

                        sequential_add(&mut part_deriv, &semi_part_deriv, curr_layer_length);

                        // update the gradients
                        perc.backward(&self.layer_inputs[layer_i - 1], z, perc_part_deriv);
                    }
                }

                for part_deriv_i in 0..curr_layer_length{
                    self.part_derivs[layer_i - 1][perc_i][part_deriv_i] = part_deriv[part_deriv_i];
                }
            }
        }

        // resets all the gradients
        for layer_i in 1..self.n_layers{
            for perc_i in 0..self.layer_dim[layer_i]{
                self.perc_layers[layer_i - 1][perc_i].update_weights();
            }
        }
    }
}

