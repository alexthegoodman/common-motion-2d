use crate::{
    data::{
        batcher::{TextGenerationBatch, TextGenerationBatcher, TrainingTextGenerationBatch},
        dataset::TextGenerationItem,
        tokenizer::Tokenizer,
    },
    model::TextGenerationModelConfig,
    training::ExperimentConfig,
};
use burn::{config::Config, nn::RotaryEncodingConfig, tensor::Int};
use burn::{data::dataloader::batcher::Batcher, tensor::cast::ToElement};
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, backend::Backend, Shape, Tensor, TensorData},
};
use std::{path::Path, sync::Arc};

use crate::{data::tokenizer::NumericalTokenizer, model::TextGenerationModel};

// training-like inference to replicate training behavior
pub fn infer_from_text_trainlike<B: Backend>(
    artifact_dir: &str,
    device: &B::Device,
    texts: Vec<String>,
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
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), 128, 512);

    // Debug: Print tokenizer vocabulary size
    println!("Tokenizer vocab size: {}", tokenizer.vocab_size());

    let rope = RotaryEncodingConfig::new(
        128 + 512, // max_sequence_length (prompt + completion)
        config.transformer.d_model,
    )
    .with_theta(10000.0);

    let model = TextGenerationModelConfig::new(
        config.transformer.clone(),
        rope,
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        128,
        512,
    )
    .init::<B>(&device);

    let model: TextGenerationModel<B> = model.load_record(record);

    // Create a batch from the input text
    let mut items = Vec::new();

    for text in texts.clone() {
        // let zeros: String = "0".repeat(512);

        // let numbers: String = (1..10)
        //     .map(|x| x.to_string())
        //     .cycle()
        //     .take(512)
        //     .collect::<Vec<_>>()
        //     .join("");

        let test_numbers = "0, 0, 361, 161, 330, -13, 
0, 2.5, 361, 161, 309, 90, 
0, 5, 361, 161, 305, 217, 
0, 15, 361, 161, 305, 217, 
0, 17.5, 361, 161, 312, 83, 
0, 20, 361, 161, 298, -22, 
1, 0, 232, 332, -17, 101, 
1, 2.5, 232, 332, 37, 86, 
1, 5, 232, 332, 50, 70, 
1, 15, 232, 332, 50, 70, 
1, 17.5, 232, 332, -5, 69, 
1, 20, 232, 332, -28, 106, 
2, 0, 149, 149, 305, -6, 
2, 2.5, 149, 149, 304, 57, 
2, 5, 149, 149, 304, 116, 
2, 15, 149, 149, 304, 116, 
2, 17.5, 149, 149, 306, 77, 
2, 20, 149, 149, 303, -11, "
            .to_string();

        //         let test_numbers = "0, 0, 361, 161, 100, 100,
        // 0, 2.5, 361, 161, 100, 100,
        // 0, 5, 361, 161, 100, 100,
        // 0, 15, 361, 161, 100, 100,
        // 0, 17.5, 361, 161, 100, 100,
        // 0, 20, 361, 161, 100, 100,
        // 1, 0, 232, 332, 100, 100,
        // 1, 2.5, 232, 332, 100, 100,
        // 1, 5, 232, 332, 100, 100,
        // 1, 15, 232, 332, 100, 100,
        // 1, 17.5, 232, 332, 100, 100,
        // 1, 20, 232, 332, 100, 100,
        // 2, 0, 149, 149, 100, 100,
        // 2, 2.5, 149, 149, 100, 100,
        // 2, 5, 149, 149, 100, 100,
        // 2, 15, 149, 149, 100, 100,
        // 2, 17.5, 149, 149, 100, 100,
        // 2, 20, 149, 149, 100, 100, "
        //             .to_string();

        let item = TextGenerationItem::new(text, test_numbers);
        println!("input item: {:?}", item);
        items.push(item);
    }

    let input: TrainingTextGenerationBatch<B> = batcher.batch(items);

    // input.completion_length = 512;

    println!("Running training inference...");

    // 2. forward_inference approach, same results
    // Run inference
    let output = model.forward_training(input);

    // Get logits from the output
    let logits = output.output;

    println!(
        "Logits length: {:?} {:?}",
        logits.dims()[0].to_usize(),
        logits.dims()[1].to_usize()
    );

    println!("Calculating probabilities...");

    // Apply softmax to get probabilities (along vocab_size dimension)
    let probs = softmax(logits, 1);

    // Get the predicted token indices
    let predicted_tokens = probs.argmax(1);

    // Process each sequence in the batch
    let sequence = predicted_tokens;

    println!("Processing sequence...");

    // let sequence = sequence.clone().reshape([sequence.dims()[1]]);

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

    println!("Predicted token ids: {:?}", predicted_token_ids);

    // Decode the tokens back to text
    let predicted_text = tokenizer.decode(&predicted_token_ids);
    println!("Generated text for input: {}", predicted_text);
}

// batch inference
// pub fn infer_from_text_batch<B: Backend>(
//     artifact_dir: &str,
//     device: &B::Device,
//     texts: Vec<String>,
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
//     let batcher = TextGenerationBatcher::new(tokenizer.clone(), 128, 512);

//     // Debug: Print tokenizer vocabulary size
//     println!("Tokenizer vocab size: {}", tokenizer.vocab_size());

//     let model = TextGenerationModelConfig::new(
//         config.transformer.clone(),
//         tokenizer.vocab_size(),
//         tokenizer.pad_token(),
//         128,
//         512,
//     )
//     .init::<B>(&device);

//     let model: TextGenerationModel<B> = model.load_record(record);

//     // Create a batch from the input text
//     let mut items = Vec::new();

//     for text in texts.clone() {
//         let item = TextGenerationItem::new(text, String::new());
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

// // sequential inference
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

    let rope = RotaryEncodingConfig::new(
        128 + 512, // max_sequence_length (prompt + completion)
        config.transformer.d_model,
    )
    .with_theta(10000.0);

    let model = TextGenerationModelConfig::new(
        config.transformer.clone(),
        rope,
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        max_prompt_len,
        max_completion_len,
    )
    .init::<B>(&device);

    let model: TextGenerationModel<B> = model.load_record(record);

    let transformer = config.transformer.init(device);
    let mut cache = transformer.new_autoregressive_cache();

    // Process each prompt
    for prompt in prompts {
        println!("Processing prompt: {}", prompt);

        // Create initial input by tokenizing the prompt
        let mut current_tokens = tokenizer.encode(&prompt, true);
        let prompt_length = current_tokens.len();

        let mut seq_length = prompt_length;

        println!("Starting sequence length: {}", seq_length);

        // Generate tokens one by one
        for _ in 0..max_new_tokens {
            // println!("Preparing batch...");

            // Prepare input batch
            let batch = prepare_inference_batch::<B>(
                &current_tokens,
                prompt_length,
                // tokenizer.clone(),
                max_new_tokens,
                device,
            );

            // println!("Running inference...");

            // Convert usize slice to Ints
            let int_tokens: Vec<i32> = current_tokens.iter().map(|&t| t as i32).collect();

            // Get model output for the current sequence
            let encoded =
                model.forward_inference_sequential(batch, &int_tokens, seq_length, &mut cache);

            println!("Encoded shape: {:?}", encoded.dims());
            println!("Existing sequence length: {}", seq_length);

            // Get predictions for the last token
            // println!("Slicing at position: {}", seq_length);

            // let last_token_logits = encoded.slice([
            //     0..1,
            //     seq_length - 1..seq_length, // This ensures we're getting a slice of size 1
            //     0..tokenizer.vocab_size(),
            // ]);

            let dim_1 = encoded.dims()[1];

            let last_token_logits = if dim_1 == 1 {
                encoded
            } else {
                encoded.slice([
                    0..1,
                    dim_1 - 1..dim_1, // This ensures we're getting a slice of size 1
                    0..tokenizer.vocab_size(),
                ])
            };

            println!("Apply temperature...");

            // Apply temperature
            let scaled_logits = if temperature != 1.0 {
                last_token_logits / temperature
            } else {
                last_token_logits
            };

            // Convert to probabilities
            let probs = softmax(scaled_logits, 2);

            let squeeze_probs: Tensor<B, 2> = probs.squeeze(0);

            // println!("Sampling from probabilities...");

            // Sample from the distribution
            let next_token = sample_from_probs::<B>(squeeze_probs.squeeze(0));

            // Append the new token
            current_tokens.push(next_token);

            // println!("Incrementing sequence length...");

            seq_length += 1;

            // Check for completion (e.g., if we generated an end token)
            // if next_token == tokenizer.end_token() {
            //     break;
            // }

            // Print current tokens every 10 tokens
            if current_tokens.len() % 10 == 0 {
                let generated_text = tokenizer.decode(&current_tokens[prompt_length..]);
                println!("Generated so far: {}", generated_text);
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
    max_new_tokens: usize,
    device: &B::Device,
) -> TextGenerationBatch<B> {
    // Create prompt tensors
    let prompt_tokens = Tensor::<B, 1, Int>::from_ints(&tokens[..prompt_length], device)
        .reshape([1, prompt_length]);
    let prompt_mask = Tensor::<B, 2, Int>::ones([1, prompt_length], device).bool();

    // Create empty completion tensors
    let max_completion_length = max_new_tokens;
    // at least 1
    // let completion_length = (tokens.len() - prompt_length).max(1);
    let completion_length = max_completion_length;
    let completion_tokens = Tensor::zeros([1, completion_length], device);
    let completion_mask = Tensor::<B, 2, Int>::zeros([1, completion_length], device).bool();

    TextGenerationBatch::new(
        prompt_tokens,
        completion_tokens,
        prompt_mask,
        completion_mask,
    )
}

// Helper function to sample from probability distribution
fn sample_from_probs<B: Backend>(probs: Tensor<B, 1>) -> usize {
    // Convert probabilities to CPU for sampling
    let probs_data = probs.to_data();
    let probs_slice: &[f32] = probs_data
        .as_slice()
        .expect("Could not get slice from tensor");

    use rand::Rng;
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen();

    let mut cumsum = 0.0_f32;
    for (idx, p) in probs_slice.iter().enumerate() {
        // Convert the tensor data to f32 explicitly
        cumsum += *p as f32;
        if r < cumsum {
            return idx;
        }
    }

    println!("Sampling failed, Fallback to last token...");
    // Fallback to last token if sampling fails
    probs_slice.len() - 1
}
