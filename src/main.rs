use data::*;
use nn::*;
use activation_functions::*;
use train::*;

mod data;
mod nn;
mod activation_functions;
mod train;

fn main() {
    let mut data_handler = DataHandler::new("./datasets/mnist.csv");

    data_handler.shuffle();

    data_handler.separate(0.8);


    // let mut nn = NeuralNetwork::new(vec![784, 100, 30, 10]);
    let mut nn = NeuralNetwork::new(vec![784, 100, 50, 10]);

    // nn.show_info();
    
    train_nn(&mut nn, &data_handler, 100, 1);

    // nn.forward(vec![0.1, 0.2]);
    nn.show_info();
    nn.save();
    
    // println!("{:?}", nn.get_outputs());
}
