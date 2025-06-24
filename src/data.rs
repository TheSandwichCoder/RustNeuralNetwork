use csv::*;
extern crate rand;

use rand::seq::SliceRandom;
use rand::thread_rng;


#[derive(Clone)]
pub struct DataVal{
    pub val: Vec<f32>,
    pub label: u8,
} 

impl DataVal{
    pub fn new(x: f32, y: f32, label: u8) -> DataVal{
        return DataVal{
            val : vec![x, y],
            label : label
        };
    }
}

pub struct DataHandler{
    pub main_data: Vec<DataVal>,
    pub main_length: usize,

    pub train_data: Vec<DataVal>,
    pub train_length: usize,

    pub test_data: Vec<DataVal>,
    pub test_length: usize,
}

impl DataHandler{
    pub fn new(filepath: &str) -> DataHandler{
        let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(filepath.to_string());

        let mut main_data: Vec<DataVal> = Vec::new();

        for result in rdr.unwrap().records(){
            
            let record: StringRecord= result.unwrap();

            let x = record[0].parse::<f32>().unwrap();
            let y = record[1].parse::<f32>().unwrap();
            let label = record[2].parse::<u8>().unwrap();

            let dataval: DataVal = DataVal::new(x/500.0, y/500.0, label);

            main_data.push(dataval);
        }

        println!("SUCCESSFULLY PARSED CSV");

        return DataHandler{main_data: main_data.clone(), main_length: main_data.len(), train_data: Vec::new(), train_length: 0, test_data: Vec::new(), test_length: 0};
    }

    pub fn shuffle(&mut self){
        let mut rng = thread_rng();
        self.main_data.shuffle(&mut rng);
    }

    pub fn separate(&mut self, sep_const: f32){
        self.train_length = (sep_const * self.main_length as f32) as usize;
        self.test_length = ((1.0 - sep_const) * self.main_length as f32) as usize;
        
        for data_i in 0..self.main_length{
            if data_i < self.train_length{
                self.train_data.push(self.main_data[data_i].clone());
            }
            else{
                self.test_data.push(self.main_data[data_i].clone());
            }
        }
    }

    pub fn show_data(&self, start: usize, end: usize){
        for data_i in start..end{
            let data = &self.main_data[data_i];
            println!("{:?} {}", data.val, data.label);
        }
    }
}
