use crate::nn::NeuralNetwork;
use crate::data::DataHandler;
use crate::activation_functions::*;

pub fn classify(outputs: &Vec<f32>) -> u8{
    let mut highest: f32 = -10000.0;
    let mut classified_i: u8 = 0;
    for output_i in 0..outputs.len(){
        if outputs[output_i] > highest{
            highest = outputs[output_i];

            classified_i = output_i as u8;
        }
    }

    return classified_i;
}

pub fn get_error(outputs: &Vec<f32>, label: u8) -> Vec<f32>{
    // let mut errors_vec = vec![0.0; outputs.len()];
    let mut actual_vec = vec![0.0; outputs.len()];

    actual_vec[label as usize] = 1.0;

    // println!("label {}", label);

    let probs = softmax(&outputs);
    let errors_vec = cross_entropy_derivative(&probs, &actual_vec);

    // for i in 0..outputs.len(){
    //     // println!("{} {}", outputs[i], actual_vec[i]);
    //     errors_vec[i] = outputs[i] - actual_vec[i] as f32;
    // }

    return errors_vec;
}

pub fn test(model: &mut NeuralNetwork, data: &DataHandler) -> (f32, f32){
    let mut n_train_correct = 0;
    let mut n_test_correct = 0;
    
    for data in &data.train_data{
        model.forward(data.val.clone());

        let model_output = model.get_outputs();

        let pred = classify(&model_output);

        if pred == data.label{
            n_train_correct += 1;
        }
    }

    for data in &data.test_data{
        model.forward(data.val.clone());

        let model_output = model.get_outputs();

        let pred = classify(&model_output);

        if pred == data.label{
            n_test_correct += 1;
        }
    }

    return (n_test_correct as f32 / data.test_length as f32, n_train_correct as f32 / data.train_length as f32);
}

pub fn train_nn(model: &mut NeuralNetwork, data_handler: &DataHandler, n_epochs: u16, epoch_batch: u16){
    for epoch in 0..n_epochs{
        for data_i in 0..data_handler.train_length{
            let data = &data_handler.train_data[data_i];

            model.forward(data.val.clone());

            let mut output = model.get_outputs();

            let prediction = classify(&output);
            
            let errors = get_error(&output, data.label);
            
            model.backward(errors);
            if data_i % 1000 == 0{
                println!("{}", data_i);
            }
        }

        if epoch % epoch_batch == 0{
            let (test_acc, train_acc) = test(model, data_handler);

            println!("epoch {} test {}% train {}%", epoch + 1, test_acc * 100.0, train_acc * 100.0);
        }
    }
}