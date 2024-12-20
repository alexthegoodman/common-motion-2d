use crate::{
    data::{
        batcher::{TextGenerationBatcher, TrainingTextGenerationBatch},
        dataset::TextGenerationItem,
        tokenizer::Tokenizer,
    },
    model::TextGenerationModelConfig,
    training::ExperimentConfig,
};
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, backend::Backend, Shape, Tensor, TensorData},
};
use std::{path::Path, sync::Arc};

use crate::{data::tokenizer::Gpt2Tokenizer, model::TextGenerationModel};

pub fn infer_from_text<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    text_items: Vec<String>,
) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Load the model
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist");

    println!("Model loaded...");

    let tokenizer = Arc::new(Gpt2Tokenizer::default());
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

    let model = TextGenerationModelConfig::new(
        config.transformer.clone(),
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        config.max_seq_length,
    )
    .init::<B>(&device);

    let model: TextGenerationModel<B> = model.load_record(record);

    // Create a batch from the input text
    let mut items = Vec::new();

    text_items.iter().for_each(|t| {
        let item = TextGenerationItem::new(t.to_string());
        items.push(item);
    });

    let input = batcher.batch(items);

    // Run inference
    let output = model.forward_training(input);

    // Get logits from the output
    let logits = output.output;

    // Apply softmax to get probabilities
    let probs = softmax(logits, 1);

    // Get the predicted token indices
    let predicted_tokens = probs.argmax(1);

    // Convert to CPU and get the raw data
    // let predicted_tokens = predicted_tokens.to_device(&B::Device::cpu());
    let predicted_data = predicted_tokens.to_data();

    // Convert bytes to Vec<usize>
    let predicted_token_ids: Vec<usize> = predicted_data
        .bytes
        .chunks(std::mem::size_of::<B::IntElem>())
        .map(|chunk| {
            let mut bytes = [0u8; 8]; // usize is 8 bytes
            bytes[..chunk.len()].copy_from_slice(chunk);
            usize::from_ne_bytes(bytes)
        })
        .collect();

    // Decode the tokens back to text
    let predicted_text = tokenizer.decode(&predicted_token_ids);
    println!("Generated text: {}", predicted_text);
}

// 3
// use crate::{
//     data::{
//         batcher::{TextGenerationBatcher, TrainingTextGenerationBatch},
//         dataset::TextGenerationItem,
//         tokenizer::Tokenizer,
//     },
//     model::TextGenerationModelConfig,
//     training::ExperimentConfig,
// };
// use burn::config::Config;
// use burn::data::dataloader::batcher::Batcher;
// use burn::{
//     module::Module,
//     record::{CompactRecorder, Recorder},
//     tensor::{activation::softmax, backend::Backend, Shape, Tensor, TensorData},
// };
// use std::{path::Path, sync::Arc};

// use crate::{data::tokenizer::Gpt2Tokenizer, model::TextGenerationModel};

// pub fn infer_from_text<B: Backend>(artifact_dir: &str, device: &B::Device, text: &str) {
//     // Load experiment configuration
//     let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
//         .expect("Config file present");

//     // Load the model
//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), device)
//         .expect("Trained model should exist");

//     println!("Model loaded...");

//     let tokenizer = Arc::new(Gpt2Tokenizer::default());
//     let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

//     let model = TextGenerationModelConfig::new(
//         config.transformer.clone(),
//         tokenizer.vocab_size(),
//         tokenizer.pad_token(),
//         config.max_seq_length,
//     )
//     .init::<B>(&device);

//     let model: TextGenerationModel<B> = model.load_record(record);

//     // Load and preprocess the text
//     let tokenizer = Gpt2Tokenizer::default();
//     let tokens = tokenizer.encode(text, true);

//     let token_ids: Vec<i32> = tokens.iter().map(|t| *t as i32).collect();
//     let token_ids = TensorData::new(token_ids, Shape::new([config.max_seq_length]));
//     let token_ids = Tensor::<B, 1>::from_data(token_ids.convert::<B::FloatElem>(), &device);

//     // Create a TrainingTextGenerationBatch
//     let item = TextGenerationItem::new(text.to_string());
//     let mut items = Vec::new();
//     items.push(item);

//     let input = batcher.batch(items); // Batch samples using the batcher

//     // Run inference
//     let output = model.forward_training(input);

//     // Convert bytes to f32 values
//     let output_data = output.output.to_data();

//     let values: Vec<f32> = output_data
//         .bytes
//         .chunks(4) // f32 is 4 bytes
//         .map(|bytes| {
//             let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
//             f32::from_le_bytes(arr)
//         })
//         .collect();

//     println!("Raw output logits: {:?}", values);

//     // let predicted_token_ids = output.output.argmax_dim(0);
//     let predicted_token_ids = output.output.argmax(1); // argmax_dim does not exist
//     let predicted_token_ids_data = predicted_token_ids.to_data();
//     let predicted_token_ids: Vec<usize> = predicted_token_ids_data
//         .to_vec() // TODO: the trait bound `usize: Element` is not satisfied
//         .expect("Couldn't convert")
//         .into_iter()
//         .map(|x: usize| x as usize)
//         .collect();
//     let predicted_text = tokenizer.decode(&predicted_token_ids);
//     println!("Generated text: {}", predicted_text);
// }

// 1
// use crate::{
//     data::{batcher::TextGenerationBatcher, tokenizer::Tokenizer},
//     model::TextGenerationModelConfig,
//     training::ExperimentConfig,
// };
// use burn::{
//     module::Module,
//     record::{CompactRecorder, Recorder},
//     tensor::{activation::softmax, backend::Backend, Shape, Tensor, TensorData},
// };
// use std::{path::Path, sync::Arc};

// use crate::{data::tokenizer::Gpt2Tokenizer, model::TextGenerationModel};

// const MAX_SEQ_LENGTH: usize = 512;

// pub fn infer_from_text<B: Backend>(
//     artifact_dir: &str,
//     device: &B::Device,
//     text: &str,
//     config: ExperimentConfig,
// ) {
//     // Load the model
//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), device)
//         .expect("Trained model should exist");

//     println!("Model loaded...");

//     let tokenizer = Arc::new(Gpt2Tokenizer::default());
//     let batcher_train = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);
//     let batcher_test = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

//     let model = TextGenerationModelConfig::new(
//         config.transformer.clone(),
//         tokenizer.vocab_size(),
//         tokenizer.pad_token(),
//         config.max_seq_length,
//     )
//     .init::<B>(&device);

//     let model: TextGenerationModel<B> = model.load_record(record);

//     // Load and preprocess the text
//     let tokenizer = Gpt2Tokenizer::default();
//     let tokens = tokenizer.encode(text, true);

//     let token_ids: Vec<i32> = tokens.iter().map(|t| *t as i32).collect();
//     let token_ids = TensorData::new(token_ids, Shape::new([MAX_SEQ_LENGTH]));
//     let token_ids = Tensor::<B, 1>::from_data(token_ids.convert::<B::FloatElem>(), &device);

//     // Run inference
//     let output = model.forward_training(token_ids); // TODO: expected struct `TrainingTextGenerationBatch<B>`, found struct `burn::tensor::Tensor<B, 1>`
//     let output_data = output.output.to_data();

//     // Convert bytes to f32 values
//     let values: Vec<f32> = output_data
//         .bytes
//         .chunks(4) // f32 is 4 bytes
//         .map(|bytes| {
//             let arr = [bytes[0], bytes[1], bytes[2], bytes[3]];
//             f32::from_le_bytes(arr)
//         })
//         .collect();

//     println!("Raw output logits: {:?}", values);

//     let predicted_token_ids = output.output.max_dim(1); // TODO: cant be -1, so what should it be?
//     let predicted_text = tokenizer.decode(
//         &predicted_token_ids
//             .to_data()
//             .to_vec() // TODO: the trait bound `usize: Element` is not satisfied
//             .expect("Couldn't convert"),
//     );
//     println!("Generated text: {}", predicted_text);
// }

// 2
// use crate::data::batcher::TrainingTextGenerationBatch;
// use crate::data::dataset::MotionDataset;
// use crate::data::tokenizer::Gpt2Tokenizer;
// use crate::data::tokenizer::Tokenizer;
// use crate::model::TextGenerationModel;
// use crate::model::TextGenerationModelConfig;

// use burn::backend::Wgpu;
// use burn::tensor::Device;
// use burn::train::ClassificationOutput;
// use burn::{
//     module::Module,
//     record::{CompactRecorder, Recorder},
//     tensor::{activation::softmax, backend::Backend, Shape, Tensor, TensorData},
// };

// type Elem = f32;

// type IBackend = burn::backend::Autodiff<Wgpu>;

// fn infer() {
//     let device = Device::<IBackend>::DiscreteGpu(0);
//     // let model_config = TextGenerationModelConfig {
//     //     transformer: burn::nn::transformer::TransformerEncoderConfig::new(128, 512, 4, 2)
//     //         .with_norm_first(true),
//     //     vocab_size: 10000,
//     //     pad_token: 0,
//     //     max_seq_length: 512,
//     // };

//     // let model = TextGenerationModel::init(&model_config, &device);

//     // // Load the trained model from disk
//     // let trained_model = burn::util::load_model("/tmp/text-generation", &device).unwrap();

//     // Load the model
//     let trained_model = CompactRecorder::new()
//         .load(format!("/tmp/text-generation/model").into(), &device)
//         .expect("Trained model should exist");

//     // Parse the input prompt
//     let prompt = "0,5,100,100,10,10\n1,5,150,100,150,110";
//     let input = parse_prompt(prompt);

//     // Generate the output
//     let output = generate_output(&trained_model, &input, &device);

//     // Convert the output into a human-readable format
//     let output_str = convert_output_to_string(&output);

//     println!("{}", output_str);
// }

// fn parse_prompt(prompt: &str) -> TrainingTextGenerationBatch<IBackend> {
//     let tokenizer = Gpt2Tokenizer::default();
//     let tokens = tokenizer.encode(prompt, true);
//     let tokens_tensor = Tensor::from_iter(tokens.iter().map(|t| *t as f32), &device);
//     let mask_pad = Tensor::ones_like(&tokens_tensor, &device);
//     let targets = Tensor::zeros_like(&tokens_tensor, &device);

//     TrainingTextGenerationBatch {
//         tokens_inputs: tokens_tensor,
//         mask_pad,
//         targets,
//     }
// }

// fn generate_output(
//     model: &TextGenerationModel<IBackend>,
//     input: &TrainingTextGenerationBatch<IBackend>,
//     device: &Device<IBackend>,
// ) -> ClassificationOutput<IBackend> {
//     model.forward_training(input.clone())
// }

// fn convert_output_to_string(output: &ClassificationOutput<IBackend>) -> String {
//     // let output_ids = output.output.argmax_dim(-1, false);
//     let output_ids = output.output.max_dim(1);
//     let tokenizer = Gpt2Tokenizer::default();
//     let output_str = tokenizer.decode(&output_ids.to_data().to_vec().expect("Couldn't convert"));
//     output_str
// }
