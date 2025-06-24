use data::*;
use nn::*;
use activation_functions::*;
use train::*;

mod data;
mod nn;
mod activation_functions;
mod train;

fn main() {
    let mut data_handler = DataHandler::new("./datasets/edited_classify14.csv");

    data_handler.shuffle();

    data_handler.separate(0.8);


    let mut nn = NeuralNetwork::new(vec![2, 50, 30, 20, 3]);

    // nn.show_info();
    
    train_nn(&mut nn, &data_handler, 2000, 10);

    // nn.forward(vec![0.1, 0.2]);
    nn.show_info();
    nn.save();
    
    // println!("{:?}", nn.get_outputs());
}
