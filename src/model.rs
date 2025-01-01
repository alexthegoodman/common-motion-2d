use crate::data::batcher::{TextGenerationBatch, TrainingTextGenerationBatch};
use burn::{
    nn::{
        attention::generate_autoregressive_mask,
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    prelude::*,
    tensor::{backend::AutodiffBackend, RangesArg},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Config)]
pub struct TextGenerationModelConfig {
    pub transformer: TransformerEncoderConfig,
    pub vocab_size: usize,
    pub pad_token: usize,
    // pub max_seq_length: usize,
    pub max_prompt_len: usize,
    pub max_completion_len: usize,
}

#[derive(Module, Debug)]
pub struct TextGenerationModel<B: Backend> {
    transformer: TransformerEncoder<B>,
    embedding_token: Embedding<B>,
    embedding_pos: Embedding<B>,
    output: Linear<B>,
    vocab_size: usize,
    pad_token: usize,
    // max_seq_length: usize,
    max_prompt_len: usize,
    max_completion_len: usize,
}

impl TextGenerationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TextGenerationModel<B> {
        let output = LinearConfig::new(self.transformer.d_model, self.vocab_size).init(device);
        let transformer = self.transformer.init(device);
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);

        // Use combined length for positional embeddings
        let total_seq_length = self.max_prompt_len + self.max_completion_len;
        let embedding_pos =
            EmbeddingConfig::new(total_seq_length, self.transformer.d_model).init(device);

        TextGenerationModel {
            transformer,
            embedding_token,
            embedding_pos,
            output,
            vocab_size: self.vocab_size,
            pad_token: self.pad_token,
            // max_seq_length: self.max_seq_length,
            max_prompt_len: self.max_prompt_len,
            max_completion_len: self.max_completion_len,
        }
    }
}
impl<B: Backend> TextGenerationModel<B> {
    // pub fn forward_training(
    //     &self,
    //     item: TrainingTextGenerationBatch<B>,
    // ) -> ClassificationOutput<B> {
    //     let [batch_size, seq_length] = item.tokens_inputs.dims();
    //     let device = &self.devices()[0];

    //     let inputs = item.tokens_inputs.to_device(device);
    //     let targets = item.targets.to_device(device);
    //     let mask_pad = item.mask_pad.to_device(device);

    //     let index_positions = Tensor::arange(0..seq_length as i64, device)
    //         .reshape([1, seq_length])
    //         .repeat_dim(0, batch_size);

    //     let embedding_positions = self.embedding_pos.forward(index_positions);
    //     let embedding_tokens = self.embedding_token.forward(inputs);
    //     let embedding = (embedding_positions + embedding_tokens) / 2;

    //     let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
    //     let encoded = self.transformer.forward(
    //         TransformerEncoderInput::new(embedding)
    //             .mask_pad(mask_pad)
    //             .mask_attn(mask_attn),
    //     );

    //     let output = self.output.forward(encoded);
    //     let output_flatten = output.reshape([batch_size * seq_length, self.vocab_size]);
    //     let targets_flatten = targets.reshape([batch_size * seq_length]);

    //     let loss = CrossEntropyLossConfig::new()
    //         .with_pad_tokens(Some(vec![self.pad_token]))
    //         .init(&output_flatten.device());
    //     let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());

    //     ClassificationOutput {
    //         loss,
    //         output: output_flatten,
    //         targets: targets_flatten,
    //     }
    // }

    pub fn forward_training(
        &self,
        item: TrainingTextGenerationBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, total_length] = item.tokens_inputs.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens_inputs.to_device(device);
        let targets = item.targets.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        // Create position indices for the entire sequence
        let index_positions = Tensor::arange(0..total_length as i64, device)
            .reshape([1, total_length])
            .repeat_dim(0, batch_size);

        // Embeddings
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        // Create causal attention mask that allows prompt tokens to attend to each other
        // but completion tokens can only attend to prompt and previous completion tokens
        let mask_attn = generate_prompt_completion_mask::<B>(
            batch_size,
            item.prompt_length,
            item.completion_length,
            device,
        );

        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );

        let output = self.output.forward(encoded);

        // We only want to compute loss on the completion portion
        let completion_start = item.prompt_length;
        let completion_end = completion_start + item.completion_length;

        let output_completion = output
            .slice([
                0..batch_size,
                completion_start..completion_end,
                0..self.vocab_size,
            ])
            .reshape([batch_size * item.completion_length, self.vocab_size]);

        let targets_completion = targets
            .slice([0..batch_size, completion_start..completion_end])
            .reshape([batch_size * item.completion_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output_completion.device());
        let loss = loss.forward(output_completion.clone(), targets_completion.clone());

        ClassificationOutput {
            loss,
            output: output_completion,
            targets: targets_completion,
        }
    }

    // pub fn forward_training(
    //     &self,
    //     item: TrainingTextGenerationBatch<B>,
    // ) -> ClassificationOutput<B> {
    //     let [batch_size, prompt_length] = item.tokens_inputs.dims();
    //     let device = &self.devices()[0];

    //     let inputs = item.tokens_inputs.to_device(device);
    //     let targets = item.targets.to_device(device);
    //     let mask_pad = item.mask_pad.to_device(device);

    //     // Create position indices for just the prompt
    //     let index_positions = Tensor::arange(0..prompt_length as i64, device)
    //         .reshape([1, prompt_length])
    //         .repeat_dim(0, batch_size);

    //     // Embeddings for prompt only
    //     let embedding_positions = self.embedding_pos.forward(index_positions);
    //     let embedding_tokens = self.embedding_token.forward(inputs);
    //     let embedding = (embedding_positions + embedding_tokens) / 2;

    //     // No need for causal mask since we're only encoding the prompt
    //     let encoded = self
    //         .transformer
    //         .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));

    //     let output = self.output.forward(encoded);

    //     // Reshape output and targets for loss computation
    //     let output_reshaped = output.reshape([batch_size * prompt_length, self.vocab_size]);
    //     let targets_reshaped = targets.reshape([batch_size * item.completion_length]);

    //     let loss = CrossEntropyLossConfig::new()
    //         .with_pad_tokens(Some(vec![self.pad_token]))
    //         .init(&output_reshaped.device());
    //     let loss = loss.forward(output_reshaped.clone(), targets_reshaped.clone());

    //     ClassificationOutput {
    //         loss,
    //         output: output_reshaped,
    //         targets: targets_reshaped,
    //     }
    // }

    pub fn forward_inference(&self, item: TrainingTextGenerationBatch<B>) -> Tensor<B, 3> {
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = &self.devices()[0];

        let inputs = item.tokens_inputs.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(inputs);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let mask_attn = generate_autoregressive_mask::<B>(batch_size, seq_length, device);
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(mask_pad)
                .mask_attn(mask_attn),
        );

        // Return logits in shape [batch_size, seq_length, vocab_size]
        self.output.forward(encoded)
    }

    // pub fn forward_inference_sequential(
    //     &self,
    //     batch: TextGenerationBatch<B>,
    //     current_tokens: &[i32], // Changed to take current tokens directly
    //     seq_length: usize,
    // ) -> Tensor<B, 3> {
    //     let [batch_size, _] = batch.prompt_tokens.dims();
    //     let device = &self.devices()[0];

    //     // Create tensor from current_tokens slice
    //     let tokens_tensor =
    //         Tensor::<B, 1, Int>::from_ints(current_tokens, device).reshape([1, seq_length]); // Reshape to [batch_size=1, seq_length]

    //     let index_positions = Tensor::arange(0..seq_length as i64, device)
    //         .reshape([1, seq_length])
    //         .repeat_dim(0, batch_size);

    //     let embedding_positions = self.embedding_pos.forward(index_positions);
    //     let embedding_tokens = self.embedding_token.forward(tokens_tensor);
    //     let embedding = (embedding_positions + embedding_tokens) / 2;

    //     // Create causal attention mask
    //     let mask_attn = generate_causal_mask::<B>(batch_size, seq_length, device);

    //     let encoded = self.transformer.forward(
    //         TransformerEncoderInput::new(embedding)
    //             .mask_pad(batch.prompt_mask)
    //             .mask_attn(mask_attn),
    //     );

    //     self.output.forward(encoded)
    // }

    pub fn forward_inference_sequential(
        &self,
        batch: TextGenerationBatch<B>,
        current_tokens: &[i32],
        seq_length: usize,
    ) -> Tensor<B, 3> {
        let device = &self.devices()[0];

        // Create tensor from current_tokens slice and reshape to [batch_size=1, seq_length]
        let tokens_tensor =
            Tensor::<B, 1, Int>::from_ints(current_tokens, device).reshape([1, seq_length]);

        // Create positional indices tensor - only use the last position for efficiency
        let last_pos = (seq_length - 1) as i64;
        let index_positions = Tensor::arange(0..seq_length as i64, device).reshape([1, seq_length]);

        // Get embeddings
        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens_tensor);
        let embedding = (embedding_positions + embedding_tokens) / 2.0;

        // Create causal attention mask for the full sequence
        let mask_attn = generate_causal_mask::<B>(1, seq_length, device);

        // Forward pass through transformer
        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(batch.prompt_mask)
                .mask_attn(mask_attn),
        );

        self.output.forward(encoded)
    }
}

impl<B: AutodiffBackend> TrainStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
    for TextGenerationModel<B>
{
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<TrainingTextGenerationBatch<B>, ClassificationOutput<B>>
    for TextGenerationModel<B>
{
    fn step(&self, item: TrainingTextGenerationBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item)
    }
}

// KV cache which makes inference faster
// pub struct KVCache<B: Backend> {
//     key_cache: Vec<Tensor<B, 4>>,
//     value_cache: Vec<Tensor<B, 4>>,
//     current_seq_len: usize,
// }

// impl<B: Backend> KVCache<B> {
//     pub fn new(
//         num_layers: usize,
//         max_seq_len: usize,
//         batch_size: usize,
//         num_heads: usize,
//         head_dim: usize,
//         device: &B::Device,
//     ) -> Self {
//         let shape = [batch_size, num_heads, max_seq_len, head_dim];
//         let key_cache = vec![Tensor::zeros(shape, device); num_layers];
//         let value_cache = vec![Tensor::zeros(shape, device); num_layers];

//         Self {
//             key_cache,
//             value_cache,
//             current_seq_len: 0,
//         }
//     }

//     pub fn update(
//         &mut self,
//         layer_idx: usize,
//         key: Tensor<B, 4>,
//         value: Tensor<B, 4>,
//         position: usize,
//     ) {
//         if position < self.key_cache[layer_idx].dims()[2] {
//             // Update specific position in cache
//             self.key_cache[layer_idx].slice_assign([.., .., position..position + 1, ..], key);
//             self.value_cache[layer_idx].slice_assign([.., .., position..position + 1, ..], value);
//         }
//         self.current_seq_len = position + 1;
//     }

//     pub fn get(&self, layer_idx: usize) -> (&Tensor<B, 4>, &Tensor<B, 4>) {
//         (&self.key_cache[layer_idx], &self.value_cache[layer_idx])
//     }
// }

fn generate_prompt_completion_mask<B: Backend>(
    batch_size: usize,
    prompt_length: usize,
    completion_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let total_length = prompt_length + completion_length;

    let rows: Tensor<B, 2, Int> = Tensor::arange(0..total_length as i64, device)
        .reshape([total_length, 1])
        .repeat_dim(1, total_length);

    let cols: Tensor<B, 2, Int> = Tensor::arange(0..total_length as i64, device)
        .reshape([1, total_length])
        .repeat_dim(0, total_length);

    // Create numeric tensors for comparison (1 for true, 0 for false)
    let prompt_rows = rows
        .clone()
        .lower_equal_elem(prompt_length as i64 - 1)
        .int();
    let prompt_cols = cols
        .clone()
        .lower_equal_elem(prompt_length as i64 - 1)
        .int();
    let prompt_mask = prompt_rows * prompt_cols;

    let completion_rows = rows.clone().greater_equal_elem(prompt_length as i64).int();
    let completion_cols = rows.greater_equal(cols).int();
    let completion_mask = completion_rows * completion_cols;

    // Convert back to boolean at the end
    let mask =
        (prompt_mask + completion_mask)
            .greater_elem(0)
            .reshape([1, total_length, total_length]);

    mask
}

// If you need a simpler version that just creates a causal mask:
fn generate_causal_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let rows = Tensor::arange(0..seq_length as i64, device)
        .reshape([seq_length, 1])
        .repeat_dim(1, seq_length);

    let cols = Tensor::arange(0..seq_length as i64, device)
        .reshape([1, seq_length])
        .repeat_dim(0, seq_length);

    // rows >= cols creates a lower triangular matrix (including diagonal)
    rows.greater_equal(cols)
        .reshape([1, seq_length, seq_length])
}
