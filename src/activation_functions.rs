use std:: cmp;

pub fn ReLu(x: f32) -> f32{
    return x.max(0.0);
}

pub fn deriv_ReLu(x: f32) -> f32{
    if x > 0.0{
        return 1.0;
    }
    return 0.0;
}

pub fn sigmoid(x: f32) -> f32{
    if x > 10.0{
        return 1.0;
    }
    else if x < -10.0{
        return 0.0;
    }

    return 1.0 / (1.0 + x.exp());
}

pub fn deriv_sigmoid(x: f32) -> f32{
    let s = sigmoid(x);
    
    return s * (1.0 - s);
}


// generously donated by ChatGPT
pub fn softmax(input: &[f32]) -> Vec<f32> {
    let max = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = input.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum).collect()
}

pub fn cross_entropy_loss(predicted: &[f32], target: &[f32]) -> f32 {
    predicted.iter()
        .zip(target.iter())
        .map(|(p, t)| -t * p.ln())
        .sum()
}

pub fn cross_entropy_derivative(softmax_output: &[f32], target: &[f32]) -> Vec<f32> {
    softmax_output.iter()
        .zip(target.iter())
        .map(|(p, t)| p - t)
        .collect()
}