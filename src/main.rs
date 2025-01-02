use burn::backend::Wgpu;
use burn::optim::decay::WeightDecayConfig;
use common_motion_2d::data::dataset::MotionDataset;
use common_motion_2d::inference::{infer_from_text, infer_from_text_trainlike};
use common_motion_2d::training::ExperimentConfig;

type Elem = f32;

type Backend = burn::backend::Autodiff<Wgpu>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(768, 3072, 12, 6)
            .with_norm_first(true)
            .with_dropout(0.1),
        burn::optim::AdamWConfig::new().with_weight_decay(1.0e-8),
    );

    common_motion_2d::training::train::<Backend, MotionDataset>(
        burn::tensor::Device::<Backend>::DiscreteGpu(0),
        MotionDataset::train().expect("Couldn't load training set"),
        MotionDataset::test().expect("Couldn't load test set"),
        config,
        "/tmp/text-generation",
    );

    // let mut prompts = Vec::new();
    // prompts.push(
    //     "0, 5, 361, 161, 305, 217, \n1, 5, 232, 332, 50, 70, \n2, 5, 149, 149, 304, 116, "
    //         .to_string(),
    // );

    // let device = burn::tensor::Device::<Backend>::DiscreteGpu(0);

    // // infer_from_text::<Backend>(
    // //     "/tmp/text-generation-e3-prompt-large",
    // //     &device,
    // //     prompts,
    // //     512,
    // //     1.0, // 1.0 should have no effect
    // // );

    // infer_from_text_trainlike::<Backend>("/tmp/text-generation-e3-prompt-large", &device, prompts);
}
