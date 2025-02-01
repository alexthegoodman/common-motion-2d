use crate::{
    data::{
        batcher::{TextGenerationBatch, TextGenerationBatcher},
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
use std::{path::Path, sync::Arc};

use crate::{data::tokenizer::NumericalTokenizer, model::TextGenerationModel};

// Sequential inference
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

    // Use the helper function to get max sequence length from your training data
    let max_sequence_length = 1024; // Adjust this value based on your dataset
    let batcher = TextGenerationBatcher::new(tokenizer.clone(), max_sequence_length);

    let rope = RotaryEncodingConfig::new(max_sequence_length, config.transformer.d_model)
        .with_theta(10000.0);

    let model = TextGenerationModelConfig::new(
        config.transformer.clone(),
        rope,
        tokenizer.vocab_size(),
        tokenizer.pad_token(),
        max_sequence_length,
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

    // Process each prompt
    for prompt in prompts {
        println!("Processing prompt: {}", prompt);

        // Create initial input by tokenizing the prompt
        let mut current_tokens: Vec<u32> =
            tokenizer.encode_inference(&prompt, true).get_ids().to_vec();
        let prompt_length = current_tokens.len();

        // Generate tokens one by one
        for _ in 0..max_new_tokens {
            // Convert current sequence to tensor
            let current_tensor: Tensor<B, 1, Int> =
                Tensor::<B, 1, Int>::from_ints(&*current_tokens, device);

            // Prepare batch and get model output
            let batch: TextGenerationBatch<B> = batcher.batch(vec![TextGenerationItem {
                sequence: tokenizer.decode_inference(&current_tokens),
            }]);

            let encoded: Tensor<B, 3> =
                model.forward_inference_sequential(&current_tokens, &mut cache);

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

            let next_token: Tensor<B, 2, Int> = sampler.sample(next_token_logits);

            // Extract the token and append to sequence
            let next_token_value: i32 =
                next_token.to_data().to_vec().expect("Could not get vec")[0];

            current_tokens.push(next_token_value as u32);

            // Print progress
            if (current_tokens.len() - prompt_length) % 10 == 0 {
                let generated_text = tokenizer.decode_inference(&current_tokens[prompt_length..]);
                println!("Generated so far: {}", generated_text);
            }
        }

        // Print final generated text
        let generated_text = tokenizer.decode_inference(&current_tokens[prompt_length..]);
        println!("Final generated text: {}", generated_text);
    }
}

// use crate::{
//     data::{
//         batcher::{TextGenerationBatch, TextGenerationBatcher, TrainingTextGenerationBatch},
//         dataset::TextGenerationItem,
//         tokenizer::Tokenizer,
//     },
//     model::TextGenerationModelConfig,
//     sampling::{Sampler, TopP},
//     training::ExperimentConfig,
// };
// use burn::{config::Config, nn::RotaryEncodingConfig, tensor::Int};
// use burn::{data::dataloader::batcher::Batcher, tensor::cast::ToElement};
// use burn::{
//     module::Module,
//     record::{CompactRecorder, Recorder},
//     tensor::{activation::softmax, backend::Backend, Shape, Tensor, TensorData},
// };
// use itertools::Itertools;
// use std::{path::Path, sync::Arc};

// use crate::{data::tokenizer::NumericalTokenizer, model::TextGenerationModel};

// // // sequential inference
// pub fn infer_from_text<B: Backend>(
//     artifact_dir: &str,
//     device: &B::Device,
//     prompts: Vec<String>,
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

//     // Use the helper function to get max lengths from your training data
//     // or use fixed values based on your known data characteristics
//     let (max_prompt_len, max_completion_len) = (128, 512); // adjust these values
//     let batcher = TextGenerationBatcher::new(tokenizer.clone(), max_prompt_len, max_completion_len);

//     let rope = RotaryEncodingConfig::new(
//         128 + 512, // max_sequence_length (prompt + completion)
//         config.transformer.d_model,
//     )
//     .with_theta(10000.0);

//     let model = TextGenerationModelConfig::new(
//         config.transformer.clone(),
//         rope,
//         tokenizer.vocab_size(),
//         tokenizer.pad_token(),
//         max_prompt_len,
//         max_completion_len,
//     )
//     .init::<B>(&device);

//     let model: TextGenerationModel<B> = model.load_record(record);

//     let transformer = config.transformer.init(device);
//     let mut cache = transformer.new_autoregressive_cache();

//     // Sampling strategy
//     let mut sampler = if temperature > 0.0 {
//         Sampler::TopP(TopP::new(0.7, 42))
//     } else {
//         Sampler::Argmax
//     };

//     // prefer more deterministic sampling for now
//     // let mut sampler = Sampler::Argmax;

//     // Process each prompt
//     for prompt in prompts {
//         println!("Processing prompt: {}", prompt);

//         // Create initial input by tokenizing the prompt
//         let mut current_tokens: tokenizers::Encoding = tokenizer.encode_inference(&prompt, true);
//         let mut current_tokens: Vec<u32> = current_tokens.get_ids().to_vec();
//         let prompt_length = current_tokens.len();

//         // Track position for attention
//         let mut input_pos = 0;

//         // Generate tokens one by one
//         for _ in 0..max_new_tokens {
//             // Convert current sequence to tensor
//             let current_tensor: Tensor<B, 2, Int> =
//                 Tensor::<B, 1, Int>::from_ints(&current_tokens[input_pos..], device)
//                     .reshape([current_tokens.len() - input_pos, 1]);

//             // Prepare batch and get model output
//             let batch: TextGenerationBatch<B> = prepare_inference_batch::<B>(
//                 &current_tensor,
//                 prompt_length,
//                 max_new_tokens,
//                 device,
//             );

//             let encoded: Tensor<B, 3> = model.forward_inference_sequential(
//                 batch,
//                 &current_tokens[input_pos..],
//                 current_tokens.len() - input_pos,
//                 &mut cache,
//             );

//             // Get logits for the next token only
//             let [batch_size, seq_len, vocab_size] = encoded.dims();
//             let next_token_logits: Tensor<B, 2> = encoded
//                 .slice([0..batch_size, seq_len - 1..seq_len])
//                 .squeeze(1);

//             // Apply temperature and sample
//             let next_token_logits = if temperature != 1.0 {
//                 softmax(next_token_logits / temperature, 1)
//             } else {
//                 softmax(next_token_logits, 1)
//             };

//             let next_token: Tensor<B, 2, Int> = sampler.sample(next_token_logits);

//             // Extract the token and append to sequence
//             let next_token_value: i32 =
//                 next_token.to_data().to_vec().expect("Could not get vec")[0];

//             // println!("Selected token: {}", next_token_value);

//             current_tokens.push(next_token_value as u32);

//             // Update input position for next iteration
//             input_pos = current_tokens.len() - 1;

//             // Print progress
//             if (current_tokens.len() - prompt_length) % 10 == 0 {
//                 let generated_text = tokenizer.decode_inference(&current_tokens[prompt_length..]);
//                 println!("Generated so far: {}", generated_text);
//             }
//         }
//     }
// }

// // NOTE: maybe easier to use new, simplified Batcher instead?
// fn prepare_inference_batch<B: Backend>(
//     tokens: &Tensor<B, 2, Int>,
//     prompt_length: usize,
//     max_new_tokens: usize,
//     device: &B::Device,
// ) -> TextGenerationBatch<B> {
//     let prompt_tokens = tokens.clone().slice([0..1, 0..prompt_length]);
//     // Mask for actual tokens we have so far
//     let current_length = tokens.dims()[1];
//     let prompt_mask = Tensor::<B, 2, Int>::ones([1, prompt_length], device).bool();

//     let completion_length = max_new_tokens;
//     let completion_tokens = Tensor::zeros([1, completion_length], device);
//     // Only mask up to current position for completion
//     let completion_mask = if current_length > prompt_length {
//         let mut mask = Tensor::<B, 2, Int>::zeros([1, completion_length], device);
//         let mut mask = mask
//             .slice([0..1, 0..(current_length - prompt_length)])
//             .ones_like();
//         mask.bool()
//     } else {
//         Tensor::<B, 2, Int>::zeros([1, completion_length], device).bool()
//     };

//     TextGenerationBatch::new(
//         prompt_tokens,
//         completion_tokens,
//         prompt_mask,
//         completion_mask,
//     )
// }
