use burn::backend::Wgpu;
use burn::optim::decay::WeightDecayConfig;
use common_motion_2d::data::dataset::MotionDataset;
use common_motion_2d::inference::infer_from_text;
use common_motion_2d::training::ExperimentConfig;

type Elem = f32;

type Backend = burn::backend::Autodiff<Wgpu>;

fn main() {
    let config = ExperimentConfig::new(
        // burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
        burn::nn::transformer::TransformerEncoderConfig::new(128, 512, 4, 2).with_norm_first(true),
        // burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-2))),
    );

    common_motion_2d::training::train::<Backend, MotionDataset>(
        burn::tensor::Device::<Backend>::DiscreteGpu(0),
        MotionDataset::train().expect("Couldn't load training set"),
        MotionDataset::test().expect("Couldn't load test set"),
        config,
        "/tmp/text-generation",
    );

    // let mut text_items = Vec::new();
    // text_items.push("0, 5, 300, 100, 305, 217".to_string());
    // text_items.push("1, 5, 200, 300, 50, 70".to_string());
    // text_items.push("2, 5, 100, 100, 304, 116".to_string());

    // let device = burn::tensor::Device::<Backend>::DiscreteGpu(0);

    // infer_from_text::<Backend>("/tmp/text-generation", &device, text_items);
}
