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