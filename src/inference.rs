use crate::{
    data::{
        batcher::{TextGenerationBatch, TextGenerationBatcher, TrainingTextGenerationBatch},
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

// pub fn infer_from_text<B: Backend>(artifact_dir: &str, device: &B::Device, texts: Vec<String>) {
//     // Load experiment configuration
//     let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
//         .expect("Config file present");

//     // Load the model
//     let record = CompactRecorder::new()
//         .load(format!("{artifact_dir}/model").into(), device)
//         .expect("Trained model should exist");

//     println!("Model loaded...");

//     let tokenizer = Arc::new(NumericalTokenizer::default());
//     let batcher = TextGenerationBatcher::new(tokenizer.clone(), 128, 512);

//     // Debug: Print tokenizer vocabulary size
//     println!("Tokenizer vocab size: {}", tokenizer.vocab_size());

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
//         let item = TextGenerationItem::new(text);
//         println!("input item: {:?}", item);
//         items.push(item);
//     }

//     let input = batcher.batch(items);

//     // 2. forward_inference approach, same results
//     // Run inference
//     let output = model.forward_inference(input);

//     // Get logits from the output - now properly shaped as [batch_size, seq_length, vocab_size]
//     let logits = output;

//     // Apply softmax to get probabilities (along vocab_size dimension)
//     let probs = softmax(logits, 2);

//     // Get the predicted token indices
//     let predicted_tokens = probs.argmax(2);

//     // Process each sequence in the batch
//     for batch_idx in 0..texts.len() {
//         let sequence = predicted_tokens
//             .clone()
//             .slice([batch_idx..batch_idx + 1, 0..predicted_tokens.dims()[1]]);
//         // let sequence = predicted_tokens.slice([batch_idx, ..]);
//         let sequence = sequence.clone().reshape([sequence.dims()[1]]);

//         let sequence_data = sequence.to_data();

//         // Convert bytes to Vec<usize>
//         let predicted_token_ids: Vec<usize> = sequence_data
//             .bytes
//             .chunks(std::mem::size_of::<B::IntElem>())
//             .map(|chunk| {
//                 let mut bytes = [0u8; 8];
//                 bytes[..chunk.len()].copy_from_slice(chunk);
//                 usize::from_ne_bytes(bytes)
//             })
//             .collect();

//         // println!(
//         //     "Predicted token ids for input {}: {:?}",
//         //     batch_idx, predicted_token_ids
//         // );

//         // Decode the tokens back to text
//         let predicted_text = tokenizer.decode(&predicted_token_ids);
//         println!("Generated text for input: {}", predicted_text);
//     }
// }

pub fn infer_from_text<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    prompts: Vec<String>,
    max_new_tokens: usize,
    temperature: f32,
) {
    // Load experiment configuration
    let config = ExperimentConfig::load(format!("{artifact_dir}/config.json").as_str())
        .expect("Config file present");

    // Load the model
    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .expect("Trained model should exist");

    println!("Model loaded...");

    let tokenizer = Arc::new(NumericalTokenizer::default());

    // Use the helper function to get max lengths from your training data
    // or use fixed values based on your known data characteristics
    let (max_prompt_len, max_completion_len) = (128, 512); // adjust these values
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), max_prompt_len, max_completion_len);

    let model = TextGenerationModelConfig::new(
        config.transformer.clone(),
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        max_prompt_len,
        max_completion_len,
    )
    .init::<B>(&device);

    let model: TextGenerationModel<B> = model.load_record(record);

    // Process each prompt
    for prompt in prompts {
        println!("Processing prompt: {}", prompt);

        // Create initial input by tokenizing the prompt
        let mut current_tokens = tokenizer.encode(&prompt, true);
        let prompt_length = current_tokens.len();

        // Generate tokens one by one
        for _ in 0..max_new_tokens {
            // Create a TextGenerationItem with empty completion
            let item = TextGenerationItem::new(
                prompt.clone(),
                String::new(), // empty completion since we're generating it
            );

            // Prepare input batch
            let batch =
                prepare_inference_batch::<B>(&current_tokens, prompt_length, &tokenizer, device);

            // Get model output for the current sequence
            let encoded = model.forward_inference(batch);

            // Get predictions for the last token
            let last_token_logits = encoded.slice([
                0..1,
                current_tokens.len() - 1..current_tokens.len(),
                0..tokenizer.vocab_size(),
            ]);

            // Apply temperature
            let scaled_logits = if temperature != 1.0 {
                last_token_logits / temperature
            } else {
                last_token_logits
            };

            // Convert to probabilities
            let probs = softmax(scaled_logits, 2);

            // Sample from the distribution
            let next_token = sample_from_probs::<B>(probs.squeeze(0).squeeze(0));

            // Append the new token
            current_tokens.push(next_token);

            // Check for completion (e.g., if we generated an end token)
            if next_token == tokenizer.end_token() {
                break;
            }
        }

        // Decode the generated sequence
        let generated_text = tokenizer.decode(&current_tokens[prompt_length..]);
        println!("Generated completion: {}", generated_text);
    }
}

// Helper function to prepare a single inference batch
fn prepare_inference_batch<B: Backend>(
    tokens: &[usize],
    prompt_length: usize,
    tokenizer: &Arc<dyn Tokenizer>,
    device: &B::Device,
) -> TextGenerationBatch<B> {
    let tensor = Tensor::from_slice(tokens, device).reshape([1, tokens.len()]);

    let mask = Tensor::ones([1, tokens.len()], device);

    TextGenerationBatch {
        tokens: tensor,
        mask_pad: mask,
    }
}

// Helper function to sample from probability distribution
fn sample_from_probs<B: Backend>(probs: Tensor<B, 1>) -> usize {
    // Convert probabilities to CPU for sampling
    let probs_data = probs.to_data();

    // Implementation depends on your random number generator choice
    // Here's a simple example using rand crate:
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();

    let mut cumsum = 0.0;
    for (idx, &p) in probs_data.as_slice().iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return idx;
        }
    }

    // Fallback to last token if sampling fails
    probs_data.len() - 1
}
