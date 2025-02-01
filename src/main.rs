use burn::backend::Wgpu;
use burn::optim::decay::WeightDecayConfig;
use common_motion_2d::data::dataset::MotionDataset;
use common_motion_2d::inference::infer_from_text;
use common_motion_2d::training::ExperimentConfig;

type Elem = f32;

type Backend = burn::backend::Autodiff<Wgpu>;

fn main() {
    // let config = ExperimentConfig::new(
    //     burn::nn::transformer::TransformerEncoderConfig::new(256, 1024, 4, 6)
    //         .with_norm_first(true)
    //         .with_dropout(0.1),
    //     burn::optim::AdamWConfig::new().with_weight_decay(1.0e-8),
    // );

    // common_motion_2d::training::train::<Backend, MotionDataset>(
    //     burn::tensor::Device::<Backend>::DiscreteGpu(0),
    //     MotionDataset::train().expect("Couldn't load training set"),
    //     MotionDataset::test().expect("Couldn't load test set"),
    //     config,
    //     "/tmp/cm2d-transformers-t-small-no-pairs",
    // );

    let mut prompts = Vec::new();
    prompts.push("0, 5, 200, 100, 402, 212, \n1, 5, 100, 200, 233, 207, ".to_string());

    let device = burn::tensor::Device::<Backend>::DiscreteGpu(0);

    infer_from_text::<Backend>(
        "/tmp/cm2d-transformers-t-small-no-pairs",
        &device,
        prompts,
        512,
        1.0, // 1.0 should have no effect
    );

    // // infer_from_text_trainlike::<Backend>("/tmp/text-generation-e8-rope", &device, prompts);
}
