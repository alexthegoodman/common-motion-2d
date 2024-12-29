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

use crate::{data::tokenizer::NumericalTokenizer, model::TextGenerationModel};

// pub fn infer_from_text<B: Backend>(
//     artifact_dir: &str,
//     device: &B::Device,
//     texts: Vec<String>,
//     max_new_tokens: usize,
//     temperature: f32,
// ) {
//     // Load experiment configuration
//     let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
//         .expect("Config file present");

//     // Load the model
//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), device)
//         .expect("Trained model should exist");

//     println!("Model loaded...");

//     let tokenizer = Arc::new(NumericalTokenizer::default());
//     let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

//     let model = TextGenerationModelConfig::new(
//         config.transformer.clone(),
//         tokenizer.vocab_size(),
//         tokenizer.pad_token(),
//         config.max_seq_length,
//     )
//     .init::<B>(&device);

//     let model: TextGenerationModel<B> = model.load_record(record);

//     for text in texts {
//         let mut current_tokens = tokenizer.encode(&text, true);
//         println!("Initial text: {}", text);
//         println!("Initial tokens: {:?}", current_tokens);

//         // Generate tokens until we hit max_new_tokens or end token
//         for i in 0..max_new_tokens {
//             // Create a batch with current sequence
//             let item = TextGenerationItem::new(tokenizer.decode(&current_tokens));
//             let input = batcher.batch(vec![item]);

//             // Get model prediction
//             let output = model.forward_training(input);
//             let logits = output.output;

//             // println!(
//             //     "Iteration {}: Current sequence length: {}",
//             //     i,
//             //     current_tokens.len()
//             // );
//             // println!("Logits shape: {:?}", logits.shape());

//             // Apply temperature
//             let scaled_logits = if temperature != 1.0 {
//                 logits.div_scalar(temperature)
//             } else {
//                 logits
//             };

//             // Get probabilities for last token only
//             // let logit_range = (
//             //     i64::try_from(current_tokens.len() - 1).unwrap(),
//             //     i64::try_from(current_tokens.len()).unwrap(),
//             // );
//             // println!("logit_range: {:?}", logit_range);
//             // let last_token_logits = scaled_logits.slice([
//             //     logit_range,
//             //     (0_i64, i64::try_from(tokenizer.vocab_size()).unwrap()),
//             // ]);

//             // Get probabilities for last token only
//             let sequence_length = i64::try_from(scaled_logits.shape().dims[0]).unwrap(); // Use logits shape instead of tokens length
//             println!("sequence_length: {:?}", sequence_length);
//             let last_token_logits = scaled_logits.slice([
//                 (
//                     sequence_length - 1, // Last position in logits
//                     sequence_length,
//                 ),
//                 (0_i64, i64::try_from(tokenizer.vocab_size()).unwrap()),
//             ]);

//             let probs = softmax(last_token_logits, 1);

//             // Sample from distribution or take argmax
//             let next_token = if temperature == 0.0 {
//                 probs.argmax(1)
//             } else {
//                 // Implement multinomial sampling here if needed
//                 probs.argmax(1) // Fallback to argmax for now
//             };

//             // Convert to token ID
//             let next_token_data = next_token.to_data();
//             let next_token_id: usize = {
//                 let mut bytes = [0u8; 8];
//                 bytes[..std::mem::size_of::<B::IntElem>()].copy_from_slice(&next_token_data.bytes);
//                 usize::from_ne_bytes(bytes)
//             };

//             println!("next token id: {}", next_token_id);

//             // Add new token to sequence
//             current_tokens.push(next_token_id);

//             // Check for end token or max length
//             if next_token_id == tokenizer.end_token()
//                 || current_tokens.len() >= config.max_seq_length
//             {
//                 break;
//             }

//             // Print intermediate result every few tokens
//             if (current_tokens.len() - tokenizer.encode(&text, true).len()) % 10 == 0 {
//                 println!("Generated so far: {}", tokenizer.decode(&current_tokens));
//             }
//         }

//         // Print final generated text
//         println!(
//             "Final generated text: {}",
//             tokenizer.decode(&current_tokens)
//         );
//         println!("--------------------");
//     }
// }

pub fn infer_from_text<B: Backend>(artifact_dir: &str, device: &B::Device, texts: Vec<String>) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Load the model
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist");

    println!("Model loaded...");

    let tokenizer = Arc::new(NumericalTokenizer::default());
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

    for text in texts.clone() {
        // Debug: Print input text and its tokens
        // println!("\nProcessing input: {}", text);
        // let tokens = tokenizer.encode(&text, true);
        // println!("Input tokens: {:?}", tokens);

        let item = TextGenerationItem::new(text);
        println!("input item: {:?}", item);
        items.push(item);
    }

    // for text in texts.clone() {
    //     // Debug: Print input text and its tokens
    //     let tokens = tokenizer.encode(&text, true);
    //     let padded_tokens = tokenizer.pad(tokens, config.max_seq_length);

    //     let item = TextGenerationItem::new(tokenizer.decode(&padded_tokens));
    //     println!("input item: {:?}", item);
    //     items.push(item);
    // }

    let input = batcher.batch(items);

    // 2. forward_inference approach, same results
    // Run inference
    let output = model.forward_inference(input);

    // Get logits from the output - now properly shaped as [batch_size, seq_length, vocab_size]
    let logits = output;

    // Apply softmax to get probabilities (along vocab_size dimension)
    let probs = softmax(logits, 2);

    // Get the predicted token indices
    let predicted_tokens = probs.argmax(2);

    // Process each sequence in the batch
    for batch_idx in 0..texts.len() {
        let sequence = predicted_tokens
            .clone()
            .slice([batch_idx..batch_idx + 1, 0..predicted_tokens.dims()[1]]);
        // let sequence = predicted_tokens.slice([batch_idx, ..]);
        let sequence = sequence.clone().reshape([sequence.dims()[1]]);

        // Convert to CPU and get the raw data
        // let sequence = sequence.to_device(&B::Device::cpu());
        let sequence_data = sequence.to_data();

        // Convert bytes to Vec<usize>
        let predicted_token_ids: Vec<usize> = sequence_data
            .bytes
            .chunks(std::mem::size_of::<B::IntElem>())
            .map(|chunk| {
                let mut bytes = [0u8; 8];
                bytes[..chunk.len()].copy_from_slice(chunk);
                usize::from_ne_bytes(bytes)
            })
            .collect();

        // println!(
        //     "Predicted token ids for input {}: {:?}",
        //     batch_idx, predicted_token_ids
        // );

        // Decode the tokens back to text
        let predicted_text = tokenizer.decode(&predicted_token_ids);
        println!("Generated text for input: {}", predicted_text);
    }
}

// pub fn infer_from_text<B: Backend>(artifact_dir: &str, device: &B::Device, texts: Vec<String>) {
//     // Load config and model setup same as before...
//     let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
//         .expect("Config file present");

//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), device)
//         .expect("Trained model should exist");

//     let tokenizer = Arc::new(NumericalTokenizer::default());
//     let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

//     let model = TextGenerationModelConfig::new(
//         config.transformer.clone(),
//         tokenizer.vocab_size(),
//         tokenizer.pad_token(),
//         config.max_seq_length,
//     )
//     .init::<B>(&device);

//     let model: TextGenerationModel<B> = model.load_record(record);

//     // Create a batch from the input text
//     let mut items = Vec::new();

//     for text in texts.clone() {
//         // Pad the input to max_seq_length with pad tokens
//         let mut item = TextGenerationItem::new(text);
//         // Optional: You can add a property to TextGenerationItem to specify desired output length
//         // item.target_length = config.max_seq_length;
//         items.push(item);
//     }

//     let input = batcher.batch(items);

//     // Run inference for the full sequence
//     let output = model.forward_inference(input);
//     let probs = softmax(output, 2);
//     let predicted_tokens = probs.argmax(2);

//     // Process each sequence in the batch
//     for batch_idx in 0..texts.len() {
//         let sequence = predicted_tokens
//             .clone()
//             .slice([batch_idx..batch_idx + 1, 0..config.max_seq_length])
//             .reshape([config.max_seq_length]);

//         let sequence_data = sequence.to_data();

//         // Convert to token IDs
//         let predicted_token_ids: Vec<usize> = sequence_data
//             .bytes
//             .chunks(std::mem::size_of::<B::IntElem>())
//             .map(|chunk| {
//                 let mut bytes = [0u8; 8];
//                 bytes[..chunk.len()].copy_from_slice(chunk);
//                 usize::from_ne_bytes(bytes)
//             })
//             .collect();

//         // Trim padding tokens if desired
//         let trimmed_tokens: Vec<usize> = predicted_token_ids
//             .into_iter()
//             .take_while(|&token| token != tokenizer.pad_token())
//             .collect();

//         let predicted_text = tokenizer.decode(&trimmed_tokens);
//         println!("Generated text for input {}: {}", batch_idx, predicted_text);
//     }
// }

// pub fn infer_from_text<B: Backend>(artifact_dir: &str, device: &B::Device, texts: Vec<String>) {
//     // Load configuration and model setup same as before...
//     let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
//         .expect("Config file present");

//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), device)
//         .expect("Trained model should exist");

//     let tokenizer = Arc::new(NumericalTokenizer::default());
//     let batcher = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

//     let model = TextGenerationModelConfig::new(
//         config.transformer.clone(),
//         tokenizer.vocab_size(),
//         tokenizer.pad_token(),
//         config.max_seq_length,
//     )
//     .init::<B>(&device);

//     let model: TextGenerationModel<B> = model.load_record(record);

//     // Process each input text
//     for text in texts {
//         let mut current_tokens = tokenizer.encode(&text, true);
//         let mut generated_sequence = current_tokens.clone();

//         // Generate tokens until we reach max length or end token
//         while generated_sequence.len() < config.max_seq_length {
//             // Create a batch with current sequence
//             let item = TextGenerationItem::new(tokenizer.decode(&current_tokens));
//             let input = batcher.batch(vec![item]);

//             // Get model predictions
//             let output = model.forward_inference_sequential_2(input);
//             let probs = softmax(output, 2);
//             let predicted_tokens = probs.argmax(2);

//             // Extract the prediction for the last position
//             let last_token = predicted_tokens
//                 .clone()
//                 .slice([
//                     0..1,
//                     predicted_tokens.dims()[1] - 1..predicted_tokens.dims()[1],
//                 ])
//                 .reshape([1])
//                 .to_data();

//             // Convert to token ID
//             let next_token = last_token
//                 .bytes
//                 .chunks(std::mem::size_of::<B::IntElem>())
//                 .next()
//                 .map(|chunk| {
//                     let mut bytes = [0u8; 8];
//                     bytes[..chunk.len()].copy_from_slice(chunk);
//                     usize::from_ne_bytes(bytes)
//                 })
//                 .expect("Should have a token");

//             // Append the new token
//             generated_sequence.push(next_token);

//             // Update current_tokens for next iteration
//             // Option 1: Use sliding window if sequence gets too long
//             if current_tokens.len() >= config.max_seq_length {
//                 current_tokens = current_tokens[1..].to_vec();
//             }
//             current_tokens.push(next_token);

//             // Optional: Break if we generate an end token
//             if next_token == tokenizer.end_token() {
//                 break;
//             }

//             // Print intermediate result every few tokens
//             if current_tokens.len() % 10 == 0 {
//                 println!("Generated so far: {}", tokenizer.decode(&current_tokens));
//             }
//         }

//         // Decode and print the final sequence
//         let generated_text = tokenizer.decode(&generated_sequence);
//         println!("Generated text: {}", generated_text);
//     }
// }
