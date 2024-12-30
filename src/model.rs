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
        let embedding_pos =
            EmbeddingConfig::new(self.max_seq_length, self.transformer.d_model).init(device);

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

    // pub fn forward_inference(&self, item: TrainingTextGenerationBatch<B>) -> Tensor<B, 3> {
    //     let [batch_size, seq_length] = item.tokens_inputs.dims();
    //     let device = &self.devices()[0];

    //     let inputs = item.tokens_inputs.to_device(device);
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

    //     // Return logits in shape [batch_size, seq_length, vocab_size]
    //     self.output.forward(encoded)
    // }

    pub fn forward_inference(&self, batch: TextGenerationBatch<B>) -> Tensor<B, 3> {
        let [batch_size, seq_length] = batch.tokens.dims();
        let device = &self.devices()[0];

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat_dim(0, batch_size);

        let embedding_positions = self.embedding_pos.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(batch.tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        // Create causal attention mask
        let mask_attn = generate_causal_mask::<B>(batch_size, seq_length, device);

        let encoded = self.transformer.forward(
            TransformerEncoderInput::new(embedding)
                .mask_pad(batch.mask_pad)
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

// Helper function to create attention mask for prompt-completion format
fn generate_prompt_completion_mask<B: Backend>(
    batch_size: usize,
    prompt_length: usize,
    completion_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let total_length = prompt_length + completion_length;
    let mut mask: Tensor<B, 2, Int> = Tensor::zeros([total_length, total_length], device);

    // Prompt tokens can attend to all prompt tokens
    mask = mask.slice([0..prompt_length, 0..prompt_length]).ones_like();

    // // Completion tokens can attend to all prompt tokens and previous completion tokens
    for i in prompt_length..total_length {
        mask = mask.slice([i - 1..i, 0..i + 1]).ones_like();
    }

    // Convert to boolean tensor and reshape to 3D
    mask.greater_equal_elem(0.5) // Convert to boolean
        .reshape([1, total_length, total_length]) // Changed to 3D
}

// Helper function for causal mask during inference
fn generate_causal_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let mut mask: Tensor<B, 2, Int> = Tensor::zeros([seq_length, seq_length], device);
    for i in 0..seq_length {
        mask = mask.slice([i - 1..i, 0..i + 1]).ones_like();
    }

    mask.greater_equal_elem(0.5)
        .reshape([1, seq_length, seq_length])
}
