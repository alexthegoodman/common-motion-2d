use crate::{
    data::{
        batcher::{TextGenerationBatch, TextGenerationBatcher, TrainingTextGenerationBatch},
        dataset::TextGenerationItem,
        tokenizer::Tokenizer,
    },
    model::TextGenerationModelConfig,
    sampling::{Sampler, TopP},
    training::ExperimentConfig,
};
use burn::{config::Config, nn::RotaryEncodingConfig, tensor::Int};
use burn::{data::dataloader::batcher::Batcher, tensor::cast::ToElement};
use burn::{
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, backend::Backend, Shape, Tensor, TensorData},
};
use itertools::Itertools;
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

    // Sampling strategy
    let mut sampler = if temperature > 0.0 {
        Sampler::TopP(TopP::new(0.7, 42))
    } else {
        Sampler::Argmax
    };

    // prefer more deterministic sampling for now
    // let mut sampler = Sampler::Argmax;

    // Process each prompt
    for prompt in prompts {
        println!("Processing prompt: {}", prompt);

        // Create initial input by tokenizing the prompt
        let mut current_tokens: tokenizers::Encoding = tokenizer.encode_inference(&prompt, true);
        let mut current_tokens: Vec<u32> = current_tokens.get_ids().to_vec();
        let prompt_length = current_tokens.len();

        // Track position for attention
        let mut input_pos = 0;

        // Generate tokens one by one
        for _ in 0..max_new_tokens {
            // Convert current sequence to tensor
            let current_tensor: Tensor<B, 2, Int> =
                Tensor::<B, 1, Int>::from_ints(&current_tokens[input_pos..], device)
                    .reshape([current_tokens.len() - input_pos, 1]);

            // Prepare batch and get model output
            let batch: TextGenerationBatch<B> = prepare_inference_batch::<B>(
                &current_tensor,
                prompt_length,
                max_new_tokens,
                device,
            );

            let encoded: Tensor<B, 3> = model.forward_inference_sequential(
                batch,
                &current_tokens[input_pos..],
                current_tokens.len() - input_pos,
                &mut cache,
            );

            // Get logits for the next token only
            let [batch_size, seq_len, vocab_size] = encoded.dims();
            let next_token_logits: Tensor<B, 2> = encoded
                .slice([0..batch_size, seq_len - 1..seq_len])
                .squeeze(1);

            // Apply temperature and sample
            let next_token_logits = if temperature != 1.0 {
                softmax(next_token_logits / temperature, 1)
            } else {
                softmax(next_token_logits, 1)
            };

            println!(
                "Top 5 logits (after softmax): {:?}",
                next_token_logits
                    .to_data()
                    .to_vec::<f32>()
                    .expect("Could not get vec")
                    .iter()
                    .enumerate()
                    .sorted_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap())
                    .take(5)
                    .collect::<Vec<_>>()
            );

            let next_token: Tensor<B, 2, Int> = sampler.sample(next_token_logits);

            // Extract the token and append to sequence
            let next_token_value: i32 =
                next_token.to_data().to_vec().expect("Could not get vec")[0];

            println!("Selected token: {}", next_token_value);

            current_tokens.push(next_token_value as u32);

            // Update input position for next iteration
            input_pos = current_tokens.len() - 1;

            // Print progress
            if (current_tokens.len() - prompt_length) % 10 == 0 {
                let generated_text = tokenizer.decode_inference(&current_tokens[prompt_length..]);
                println!("Generated so far: {}", generated_text);
            }

            // Optional: Check for end token
            // if next_token_value as u32 == tokenizer.end_token() {
            //     break;
            // }
        }

        // Final output
        // let generated_text = tokenizer.decode(&current_tokens[prompt_length..]);
        // println!("Final completion: {}", generated_text);
    }
}

// Helper function to prepare a single inference batch
fn prepare_inference_batch<B: Backend>(
    // tokens: &[usize],
    tokens: &Tensor<B, 2, Int>,
    prompt_length: usize,
    max_new_tokens: usize,
    device: &B::Device,
) -> TextGenerationBatch<B> {
    // Create prompt tensors
    // let prompt_tokens = Tensor::<B, 1, Int>::from_ints(&tokens[..prompt_length], device)
    //     .reshape([1, prompt_length]);
    let prompt_tokens = tokens.clone().slice([0..1, 0..prompt_length]);
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
