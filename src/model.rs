use crate::data::batcher::TextGenerationBatch;
use burn::{
    nn::{
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    prelude::*,
    tensor::{backend::AutodiffBackend, Int, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};
use nn::{
    transformer::TransformerEncoderAutoregressiveCache, RotaryEncoding, RotaryEncodingConfig,
};

#[derive(Config)]
pub struct TextGenerationModelConfig {
    pub transformer: TransformerEncoderConfig,
    pub rotary: RotaryEncodingConfig,
    pub vocab_size: usize,
    pub pad_token: usize,
    pub max_sequence_length: usize, // Single max length for the entire sequence
}

#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    output: Linear<B>,
    rotary: RotaryEncoding<B>,
    vocab_size: usize,
    pad_token: usize,
    max_sequence_length: usize,
}

impl TextGenerationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextGenerationModel<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);

        let rotary = self.rotary.init(device);

        TextGenerationModel {
            transformer,
            embedding_token,
            output,
            rotary,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            max_sequence_length: self.max_sequence_length,
        }
    }
}

impl<B: Backend> TextGenerationModel<B> {
    pub fn forward_training(&self, batch: TextGenerationBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = batch.tokens.dims();
        let device = &self.devices()[0];

        // Move tensors to the correct device
        let tokens = batch.tokens.to_device(device);
        let targets = batch.targets.to_device(device);
        let mask_pad = batch.mask_pad.to_device(device);

        // Token embeddings
        let embeddings = self.embedding_token.forward(tokens.clone());

        // Apply rotary encoding
        let embeddings = embeddings.unsqueeze::<4>();
        let embeddings_with_rope = self.rotary.forward(embeddings);
        let embeddings_with_rope = embeddings_with_rope.squeeze::<3>(0);

        // Generate causal attention mask
        let mask_attn = generate_autoregressive_mask(batch_size, seq_length, device);

        // Forward pass through the transformer
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embeddings_with_rope)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );

        // Output logits
        let output = self.output.forward(encoded);

        // Reshape for loss calculation
        let output = output.reshape([batch_size * seq_length, self.vocab_size]);
        let targets = targets.reshape([batch_size * seq_length]);

        // Compute loss
        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output.device());
        let loss = loss.forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }

    pub fn forward_inference_sequential(
        &self,
        current_tokens: &[u32],
        cache: &mut TransformerEncoderAutoregressiveCache<B>,
    ) -> Tensor<B, 3> {
        let device = &self.devices()[0];

        // Process only the newest token
        let last_token = &current_tokens[current_tokens.len() - 1..];
        let tokens_tensor = Tensor::<B, 1, Int>::from_ints(last_token, device).reshape([1, 1]);

        // Token embeddings
        let embeddings = self.embedding_token.forward(tokens_tensor.clone());

        // Apply rotary encoding
        let embeddings = embeddings.unsqueeze::<4>();
        let embeddings_with_rope = self.rotary.forward(embeddings);
        let embeddings_with_rope = embeddings_with_rope.squeeze::<3>(0);

        // Generate causal mask for the new token
        let mask_attn = generate_autoregressive_mask(1, 1, device);

        // Forward pass through the transformer with cache
        let encoded = self.transformer.forward_autoregressive_inference(
            TransformerEncoderInput::new(embeddings_with_rope).mask_attn(mask_attn),
            cache,
        );

        // Output logits
        self.output.forward(encoded)
    }
}

impl<B: AutodiffBackend> TrainStep<TextGenerationBatch<B>, ClassificationOutput<B>>
    for TextGenerationModel<B>
{
    fn step(&self, batch: TextGenerationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let output = self.forward_training(batch);
        let grads = output.loss.backward();

        TrainOutput::new(self, grads, output)
    }
}

impl<B: Backend> ValidStep<TextGenerationBatch<B>, ClassificationOutput<B>>
    for TextGenerationModel<B>
{
    fn step(&self, batch: TextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(batch)
    }
}

// use crate::data::batcher::{TextGenerationBatch, TrainingTextGenerationBatch};
// use burn::{
//     nn::{
//         attention::generate_autoregressive_mask,
//         loss::CrossEntropyLossConfig,
//         transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
//         Embedding, EmbeddingConfig, Linear, LinearConfig,
//     },
//     prelude::*,
//     tensor::{backend::AutodiffBackend, RangesArg},
//     train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
// };
// use nn::{
//     transformer::TransformerEncoderAutoregressiveCache, RotaryEncoding, RotaryEncodingConfig,
// };

// #[derive(Config)]
// pub struct TextGenerationModelConfig {
//     pub transformer: TransformerEncoderConfig,
//     pub rotary: RotaryEncodingConfig,
//     pub vocab_size: usize,
//     pub pad_token: usize,
//     // pub max_seq_length: usize,
//     pub max_prompt_len: usize,
//     pub max_completion_len: usize,
// }

// #[derive(Module, Debug)]
// pub struct TextGenerationModel<B: Backend> {
//     transformer: TransformerEncoder<B>,
//     embedding_token: Embedding<B>,
//     // embedding_pos: Embedding<B>,
//     output: Linear<B>,
//     rotary: RotaryEncoding<B>,
//     vocab_size: usize,
//     pad_token: usize,
//     // max_seq_length: usize,
//     max_prompt_len: usize,
//     max_completion_len: usize,
// }

// impl TextGenerationModelConfig {
//     pub fn init<B: Backend>(&self, device: &B::Device) -> TextGenerationModel<B> {
//         let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init(device);
//         let transformer = self.transformer.init(device);
//         let embedding_token =
//             EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);

//         let rotary = self.rotary.init(device);

//         TextGenerationModel {
//             transformer,
//             embedding_token,
//             output,
//             rotary,
//             vocab_size: self.vocab_size,
//             pad_token: self.pad_token,
//             max_prompt_len: self.max_prompt_len,
//             max_completion_len: self.max_completion_len,
//         }
//     }
// }
// impl<B: Backend> TextGenerationModel<B> {
//     pub fn forward_training(
//         &self,
//         item: TrainingTextGenerationBatch<B>,
//     ) -> ClassificationOutput<B> {
//         let [batch_size, total_length] = item.tokens_inputs.dims();
//         let device = &self.devices()[0];

//         let inputs = item.tokens_inputs.to_device(device);
//         let targets = item.targets.to_device(device);
//         let mask_pad = item.mask_pad.to_device(device);

//         // Token embeddings only - we don't need separate positional embeddings anymore
//         let embedding = self.embedding_token.forward(inputs);

//         let embedding = embedding.unsqueeze::<4>();

//         let embedding_with_rope = self.rotary.forward(embedding);

//         let embedding_with_rope = embedding_with_rope.squeeze::<3>(0);

//         // Create causal attention mask
//         let mask_attn = generate_prompt_completion_mask::<B>(
//             batch_size,
//             item.prompt_length,
//             item.completion_length,
//             device,
//         );

//         let encoded = self.transformer.forward(
//             TransformerEncoderInput::new(embedding_with_rope)
//                 .mask_pad(mask_pad)
//                 .mask_attn(mask_attn),
//         );

//         let output = self.output.forward(encoded);

//         // Completion portion processing remains the same
//         let completion_start = item.prompt_length;
//         let completion_end = completion_start + item.completion_length;

//         let output_completion = output
//             .slice([
//                 0..batch_size,
//                 completion_start..completion_end,
//                 0..self.vocab_size,
//             ])
//             .reshape([batch_size * item.completion_length, self.vocab_size]);

//         let targets_completion = targets
//             .slice([0..batch_size, completion_start..completion_end])
//             .reshape([batch_size * item.completion_length]);

//         let loss = CrossEntropyLossConfig::new()
//             .with_pad_tokens(Some(vec![self.pad_token]))
//             .init(&output_completion.device());
//         let loss = loss.forward(output_completion.clone(), targets_completion.clone());

//         ClassificationOutput {
//             loss,
//             output: output_completion,
//             targets: targets_completion,
//         }
//     }

//     pub fn forward_inference_sequential(
//         &self,
//         batch: TextGenerationBatch<B>,
//         current_tokens: &[u32],
//         seq_length: usize,
//         cache: &mut TransformerEncoderAutoregressiveCache<B>,
//     ) -> Tensor<B, 3> {
//         let device = &self.devices()[0];

//         // We only need to process the newest token for inference
//         let last_token = &current_tokens[current_tokens.len() - 1..];
//         let tokens_tensor = Tensor::<B, 1, Int>::from_ints(last_token, device).reshape([1, 1]);

//         // Get embeddings for just the new token
//         let embedding = self.embedding_token.forward(tokens_tensor);
//         let embedding = embedding.unsqueeze::<4>();
//         let embedding_with_rope = self.rotary.forward(embedding);
//         let embedding_with_rope = embedding_with_rope.squeeze::<3>(0);

//         // For sequential inference, we only need the mask for the new position
//         // The cache handles previous positions
//         let mask_attn = generate_causal_mask::<B>(1, 1, device);

//         let encoded = self.transformer.forward_autoregressive_inference(
//             TransformerEncoderInput::new(embedding_with_rope)
//                 .mask_pad(batch.prompt_mask.slice([0..1, seq_length - 1..seq_length])) // Only use mask for new position
//                 .mask_attn(mask_attn),
//             cache,
//         );

//         self.output.forward(encoded)
//     }
// }

// impl<B: AutodiffBackend> TrainStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
//     for TextGenerationModel<B>
// {
//     fn step(&self, item: TrainingTextGenerationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
//         let item = self.forward_training(item);
//         let grads = item.loss.backward();

//         TrainOutput::new(self, grads, item)
//     }
// }

// impl<B: Backend> ValidStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
//     for TextGenerationModel<B>
// {
//     fn step(&self, item: TrainingTextGenerationBatch<B>) -> ClassificationOutput<B> {
//         self.forward_training(item)
//     }
// }

// fn generate_prompt_completion_mask<B: Backend>(
//     batch_size: usize,
//     prompt_length: usize,
//     completion_length: usize,
//     device: &B::Device,
// ) -> Tensor<B, 3, Bool> {
//     let total_length = prompt_length + completion_length;

//     let rows: Tensor<B, 2, Int> = Tensor::arange(0..total_length as i64, device)
//         .reshape([total_length, 1])
//         .repeat_dim(1, total_length);

//     let cols: Tensor<B, 2, Int> = Tensor::arange(0..total_length as i64, device)
//         .reshape([1, total_length])
//         .repeat_dim(0, total_length);

//     // Create numeric tensors for comparison (1 for true, 0 for false)
//     let prompt_rows = rows
//         .clone()
//         .lower_equal_elem(prompt_length as i64 - 1)
//         .int();
//     let prompt_cols = cols
//         .clone()
//         .lower_equal_elem(prompt_length as i64 - 1)
//         .int();
//     let prompt_mask = prompt_rows * prompt_cols;

//     let completion_rows = rows.clone().greater_equal_elem(prompt_length as i64).int();
//     let completion_cols = rows.greater_equal(cols).int();
//     let completion_mask = completion_rows * completion_cols;

//     // Convert back to boolean at the end
//     let mask =
//         (prompt_mask + completion_mask)
//             .greater_elem(0)
//             .reshape([1, total_length, total_length]);

//     mask
// }

// // If you need a simpler version that just creates a causal mask:
// fn generate_causal_mask<B: Backend>(
//     batch_size: usize,
//     seq_length: usize,
//     device: &B::Device,
// ) -> Tensor<B, 3, Bool> {
//     let rows = Tensor::arange(0..seq_length as i64, device)
//         .reshape([seq_length, 1])
//         .repeat_dim(1, seq_length);

//     let cols = Tensor::arange(0..seq_length as i64, device)
//         .reshape([1, seq_length])
//         .repeat_dim(0, seq_length);

//     // rows >= cols creates a lower triangular matrix (including diagonal)
//     rows.greater_equal(cols)
//         .reshape([1, seq_length, seq_length])
// }
