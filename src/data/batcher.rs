use super::{
    dataset::{MotionDataset, TextGenerationItem},
    tokenizer::Tokenizer,
};
use burn::{
    data::dataloader::{batcher::Batcher, Dataset},
    nn::attention::generate_padding_mask,
    prelude::*,
};
use derive_new::new;
use std::sync::Arc;

#[derive(Clone, new)]
pub struct TextGenerationBatcher {
    tokenizer: Arc<dyn Tokenizer>,
    max_sequence_length: usize, // Single max length for the entire sequence
}

#[derive(Debug, Clone, new)]
pub struct TextGenerationBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,    // Combined input tokens
    pub targets: Tensor<B, 2, Int>,   // Shifted targets
    pub mask_pad: Tensor<B, 2, Bool>, // Padding mask
    pub sequence_length: usize,       // Actual sequence length for this batch
}

impl<B: Backend> Batcher<TextGenerationItem, TextGenerationBatch<B>> for TextGenerationBatcher {
    fn batch(&self, items: Vec<TextGenerationItem>) -> TextGenerationBatch<B> {
        let mut tokenized_sequences = Vec::with_capacity(items.len());

        // Tokenize each sequence
        for item in items {
            let tokens = self.tokenizer.encode(&item.sequence, true);
            tokenized_sequences.push(tokens);
        }

        // Generate padding mask and truncate/pad sequences to max_sequence_length
        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokenized_sequences,
            Some(self.max_sequence_length),
            &B::Device::default(),
        );

        // Extract tokens and mask from the padding mask result
        let tokens = mask.tensor.clone();
        let mask_pad = mask.mask.clone();

        // Create targets by shifting the tokens
        let targets = tokens
            .clone()
            .slice([0..tokens.dims()[0], 1..tokens.dims()[1]]);

        // Truncate the last token of the input sequence to align with targets
        let tokens = tokens
            .clone()
            .slice([0..tokens.dims()[0], 0..tokens.dims()[1] - 1]);

        // Truncate the mask to match the input sequence length
        let mask_pad = mask_pad
            .clone()
            .slice([0..mask_pad.dims()[0], 0..mask_pad.dims()[1] - 1]);

        // Get the actual sequence length (excluding padding)
        let sequence_length = tokens.dims()[1];

        TextGenerationBatch {
            tokens,
            targets,
            mask_pad,
            sequence_length,
        }
    }
}

// Helper function to get max sequence length from a dataset
impl TextGenerationBatcher {
    pub fn get_max_sequence_length(
        dataset: &MotionDataset,
        tokenizer: &Arc<dyn Tokenizer>,
    ) -> usize {
        let mut max_length = 0;

        for i in 0..dataset.len() {
            if let Some(item) = dataset.get(i) {
                let tokens = tokenizer.encode(&item.sequence, true);
                max_length = max_length.max(tokens.len());
            }
        }

        max_length
    }
}

// use super::{
//     dataset::{MotionDataset, TextGenerationItem},
//     tokenizer::Tokenizer,
// };
// use burn::data::dataloader::Dataset;
// use burn::{data::dataloader::batcher::Batcher, nn::attention::generate_padding_mask, prelude::*};
// use derive_new::new;
// use std::sync::Arc;

// #[derive(Clone, new)]
// pub struct TextGenerationBatcher {
//     tokenizer: Arc<dyn Tokenizer>,
//     max_prompt_length: usize,
//     max_completion_length: usize,
// }

// #[derive(Debug, Clone, new)]
// pub struct TextGenerationBatch<B: Backend> {
//     pub prompt_tokens: Tensor<B, 2, Int>,
//     pub completion_tokens: Tensor<B, 2, Int>,
//     pub prompt_mask: Tensor<B, 2, Bool>,
//     pub completion_mask: Tensor<B, 2, Bool>,
// }

// #[derive(Debug, Clone, new)]
// pub struct TrainingTextGenerationBatch<B: Backend> {
//     pub tokens_inputs: Tensor<B, 2, Int>,
//     pub targets: Tensor<B, 2, Int>,
//     pub mask_pad: Tensor<B, 2, Bool>,
//     pub prompt_length: usize, // Store actual prompt length for this batch
//     pub completion_length: usize, // Store actual completion length for this batch
// }

// impl<B: Backend> Batcher<TextGenerationItem, TextGenerationBatch<B>> for TextGenerationBatcher {
//     fn batch(&self, items: Vec<TextGenerationItem>) -> TextGenerationBatch<B> {
//         let mut prompt_tokens_list = Vec::with_capacity(items.len());
//         let mut completion_tokens_list = Vec::with_capacity(items.len());

//         for item in items {
//             prompt_tokens_list.push(self.tokenizer.encode(&item.prompt, true));
//             completion_tokens_list.push(self.tokenizer.encode(&item.completion, true));
//         }

//         let prompt_mask = generate_padding_mask(
//             self.tokenizer.pad_token(),
//             prompt_tokens_list,
//             Some(self.max_prompt_length),
//             &B::Device::default(),
//         );

//         let completion_mask = generate_padding_mask(
//             self.tokenizer.pad_token(),
//             completion_tokens_list,
//             Some(self.max_completion_length),
//             &B::Device::default(),
//         );

//         TextGenerationBatch {
//             prompt_tokens: prompt_mask.tensor,
//             completion_tokens: completion_mask.tensor,
//             prompt_mask: prompt_mask.mask,
//             completion_mask: completion_mask.mask,
//         }
//     }
// }

// impl<B: Backend> Batcher<TextGenerationItem, TrainingTextGenerationBatch<B>>
//     for TextGenerationBatcher
// {
//     fn batch(&self, items: Vec<TextGenerationItem>) -> TrainingTextGenerationBatch<B> {
//         let batch: TextGenerationBatch<B> = self.batch(items);
//         let device = &B::Device::default();

//         // Get actual lengths from this batch
//         let [batch_size, prompt_length] = batch.prompt_tokens.dims();
//         let [_, completion_length] = batch.completion_tokens.dims();

//         // Combine prompt and completion tokens
//         let tokens = Tensor::cat(
//             vec![
//                 batch.prompt_tokens.clone(),
//                 batch
//                     .completion_tokens
//                     .clone()
//                     .slice([0..batch_size, 0..completion_length - 1]),
//             ],
//             1,
//         );

//         // Create targets by shifting completion tokens right and adding prompt tokens
//         let targets = Tensor::cat(
//             vec![
//                 batch.prompt_tokens,
//                 batch
//                     .completion_tokens
//                     .slice([0..batch_size, 1..completion_length]),
//             ],
//             1,
//         );

//         // Combine masks similarly
//         let mask_pad = Tensor::cat(
//             vec![
//                 batch.prompt_mask,
//                 batch
//                     .completion_mask
//                     .slice([0..batch_size, 0..completion_length - 1]),
//             ],
//             1,
//         );

//         TrainingTextGenerationBatch::new(
//             tokens,
//             targets,
//             mask_pad,
//             prompt_length,
//             completion_length - 1, // -1 because we removed last token for shifting
//         )
//     }
// }

// // Helper function to get max lengths from a dataset
// impl TextGenerationBatcher {
//     pub fn get_max_lengths(
//         dataset: &MotionDataset,
//         tokenizer: &Arc<dyn Tokenizer>,
//     ) -> (usize, usize) {
//         let mut max_prompt_length = 0;
//         let mut max_completion_length = 0;

//         for i in 0..dataset.len() {
//             if let Some(item) = dataset.get(i) {
//                 let prompt_tokens = tokenizer.encode(&item.prompt, true);
//                 let completion_tokens = tokenizer.encode(&item.completion, true);

//                 max_prompt_length = max_prompt_length.max(prompt_tokens.len());
//                 max_completion_length = max_completion_length.max(completion_tokens.len());
//             }
//         }

//         (max_prompt_length, max_completion_length)
//     }
// }
