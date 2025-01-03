use crate::{
    data::batcher::TextGenerationBatcher,
    data::dataset::TextGenerationItem,
    data::tokenizer::{NumericalTokenizer, Tokenizer},
    model::TextGenerationModelConfig,
};
use burn::{
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{transform::SamplerDataset, Dataset},
    },
    lr_scheduler::{self, constant::ConstantLr, noam::NoamLrSchedulerConfig},
    nn::transformer::TransformerEncoderConfig,
    optim::{AdamConfig, AdamWConfig},
    prelude::*,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CudaMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use nn::RotaryEncodingConfig;
use std::sync::Arc;

#[derive(Config)]
pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamWConfig,
    #[config(default = 512)]
    // #[config(default = 1024)]
    pub max_seq_length: usize,
    #[config(default = 2)]
    pub batch_size: usize,
    #[config(default = 50)]
    pub num_epochs: usize,
}

pub fn train<B: AutodiffBackend, D: Dataset<TextGenerationItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(NumericalTokenizer::default());
    // let (max_prompt_len, max_completion_len) =
    //     TextGenerationBatcher::get_max_lengths(&dataset_train, &tokenizer);
    let batcher_train = TextGenerationBatcher::new(tokenizer.clone(), 128, 512);
    let batcher_test = TextGenerationBatcher::new(tokenizer.clone(), 128, 512);

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
        // config.max_seq_length,
        128,
        512,
    )
    .init::<B>(&device);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(4)
        .shuffle(42)
        // .build(SamplerDataset::new(dataset_train, 10_000));
        // .build(SamplerDataset::new(dataset_train, 4326));
        .build(SamplerDataset::new(dataset_train, 950));

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(4)
        .shuffle(42)
        // .build(SamplerDataset::new(dataset_test, 1000));
        // .build(SamplerDataset::new(dataset_test, 1050));
        .build(SamplerDataset::new(dataset_test, 45));

    // let accum = 1; // Effective batch size = 6 * 6 = 32. 32 is the "best maximum"
    let accum = config.batch_size;
    let optim = config.optimizer.init();
    // let lr_scheduler = NoamLrSchedulerConfig::new(0.01 / accum as f64)
    //     .with_warmup_steps(6000)
    //     // .with_warmup_steps(2000)
    //     .with_model_size(config.transformer.d_model)
    //     .init();
    // let lr_scheduler = ConstantLr::new(0.00000001); // no learning noted
    // let lr_scheduler = ConstantLr::new(0.01); // spike followed by decrease
    // let lr_scheduler = ConstantLr::new(0.00001); // fast learning, quick to stabilize loss at around 1.57
    let lr_scheduler = ConstantLr::new(0.00005); // better for batch size of 2, similar to 0.00001 at batch size of 1
                                                 // let lr_scheduler = ConstantLr::new(0.00007);
                                                 // let lr_scheduler = ConstantLr::new(0.0001); // slightly, odd
                                                 // let lr_scheduler = ConstantLr::new(0.000001); // slightly slower, but quick to stabilize loss at around 1.73
                                                 // let lr_scheduler = ConstantLr::new(0.0000001); // no learning

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_valid_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .grads_accumulation(accum)
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optim, lr_scheduler);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config.save(format!("{artifact_dir}/config.json")).unwrap();

    DefaultRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
