use data::*;
use nn::*;
use activation_functions::*;
use train::*;
use functions::*;

mod data;
mod nn;
mod activation_functions;
mod train;
mod functions;

fn main() {
    // let mut data_handler = DataHandler::new("./datasets/mnist_letters.csv");

    // data_handler.shuffle();

    // data_handler.separate(0.8);

    let mut nn = NeuralNetwork::load("neuralnets/nn_read.txt");

    // train_nn(&mut nn, &data_handler, 100, 100);

    // nn.save();

    // println!("NN SAVED");

    let mut data_handler = DataHandler::new("./datasets/mnist_letters_default.csv");

    data_handler.shuffle();

    data_handler.separate(0.8);


    // let mut nn = NeuralNetwork::new(vec![784, 100, 30, 10]);
    // let mut nn = NeuralNetwork::new(vec![784, 128, 64, 32, 26]);

    // nn.show_info();
    
    train_nn(&mut nn, &data_handler, 100, 10);

    // let mut data_handler2 = DataHandler::new("./datasets/mnist_letters.csv");

    // data_handler2.shuffle();

    // data_handler2.separate(0.8);

    // println!("training with noise");
    // train_nn(&mut nn, &data_handler2, 100, 100);

    // nn.forward(vec![0.1, 0.2]);
    // nn.show_info();
    nn.save();
    
    // println!("{:?}", nn.get_outputs());
}
