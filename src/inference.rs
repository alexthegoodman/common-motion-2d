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

pub fn infer_from_text<B: Backend>(artifact_dir: &str, device: &B::Device, texts: Vec<String>) {
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

    // Debug: Print tokenizer vocabulary size
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());

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

    // text_items.iter().for_each(|t| {
    //     let item = TextGenerationItem::new(t.to_string());
    //     items.push(item);
    // });
    for text in texts.clone() {
        // Debug: Print input text and its tokens
        println!("\nProcessing input: {}", text);
        let tokens = tokenizer.encode(&text, true);
        println!("Input tokens: {:?}", tokens);

        let item = TextGenerationItem::new(text);
        items.push(item);
    }

    let input = batcher.batch(items);

    // println!("input: {:?}", input);

    // Run inference
    let output = model.forward_training(input);

    println!(
        "classification output targets {:?}",
        output.targets.to_data()
    );

    // Get logits from the output
    let logits = output.output;

    println!("logits {:?}", logits);

    // Apply softmax to get probabilities
    let probs = softmax(logits, 1);

    println!("probs: {:?}", probs);

    // Get the predicted token indices
    let predicted_tokens = probs.argmax(1);

    println!("predicted_tokens {:?}", predicted_tokens);

    let predicted_data = predicted_tokens.to_data();

    // Debug: Print shape information
    println!("\nPredicted shape: {:?}", predicted_data.shape);
    println!("\nPredicted: {:?}", predicted_data);

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

    // Debug: Print predicted tokens before decoding
    println!("Predicted token ids: {:?}", predicted_token_ids);

    // Decode the tokens back to text
    let predicted_text = tokenizer.decode(&predicted_token_ids);
    println!("Generated text: {}", predicted_text);

    // forward_inference approach, same results
    // // Run inference
    // let output = model.forward_inference(input);

    // // Get logits from the output - now properly shaped as [batch_size, seq_length, vocab_size]
    // let logits = output;

    // // Apply softmax to get probabilities (along vocab_size dimension)
    // let probs = softmax(logits, 2);

    // // Get the predicted token indices
    // let predicted_tokens = probs.argmax(2);

    // // Process each sequence in the batch
    // for batch_idx in 0..texts.len() {
    //     let sequence = predicted_tokens
    //         .clone()
    //         .slice([batch_idx..batch_idx + 1, 0..predicted_tokens.dims()[1]]);
    //     // let sequence = predicted_tokens.slice([batch_idx, ..]);
    //     let sequence = sequence.clone().reshape([sequence.dims()[1]]);

    //     // Convert to CPU and get the raw data
    //     // let sequence = sequence.to_device(&B::Device::cpu());
    //     let sequence_data = sequence.to_data();

    //     // Convert bytes to Vec<usize>
    //     let predicted_token_ids: Vec<usize> = sequence_data
    //         .bytes
    //         .chunks(std::mem::size_of::<B::IntElem>())
    //         .map(|chunk| {
    //             let mut bytes = [0u8; 8];
    //             bytes[..chunk.len()].copy_from_slice(chunk);
    //             usize::from_ne_bytes(bytes)
    //         })
    //         .collect();

    //     // Decode the tokens back to text
    //     let predicted_text = tokenizer.decode(&predicted_token_ids);
    //     println!("Generated text for input {}: {}", batch_idx, predicted_text);
    // }
}
