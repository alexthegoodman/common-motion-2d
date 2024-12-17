use burn::backend::Wgpu;
use burn::optim::decay::WeightDecayConfig;
use common_motion_2d::data::dataset::MotionDataset;
use common_motion_2d::training::ExperimentConfig;

type Elem = f32;

type Backend = burn::backend::Autodiff<Wgpu>;

fn main() {
    let config = ExperimentConfig::new(
        burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
            .with_norm_first(true),
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    common_motion_2d::training::train::<Backend, MotionDataset>(
        burn::tensor::Device::<Backend>::DiscreteGpu(0),
        MotionDataset::train().expect("Couldn't load training set"),
        MotionDataset::test().expect("Couldn't load test set"),
        config,
        "/tmp/text-generation",
    );
}
