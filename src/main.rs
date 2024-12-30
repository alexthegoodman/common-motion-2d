use burn::backend::Wgpu;
use burn::optim::decay::WeightDecayConfig;
use common_motion_2d::data::dataset::MotionDataset;
use common_motion_2d::inference::infer_from_text;
use common_motion_2d::training::ExperimentConfig;

type Elem = f32;

type Backend = burn::backend::Autodiff<Wgpu>;

fn main() {
    // let config = ExperimentConfig::new(
    //     // burn::nn::transformer::TransformerEncoderConfig::new(384, 1536, 12, 6)
    //     //     .with_norm_first(true),
    //     // burn::nn::transformer::TransformerEncoderConfig::new(128, 512, 4, 4).with_norm_first(true),
    //     // burn::nn::transformer::TransformerEncoderConfig::new(768, 3072, 16, 6)
    //     //     .with_norm_first(true),
    //     burn::nn::transformer::TransformerEncoderConfig::new(768, 3072, 12, 6)
    //         .with_norm_first(true),
    //     // burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    //     burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-8))),
    //     // burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-2))),
    // );

    // common_motion_2d::training::train::<Backend, MotionDataset>(
    //     burn::tensor::Device::<Backend>::DiscreteGpu(0),
    //     MotionDataset::train().expect("Couldn't load training set"),
    //     MotionDataset::test().expect("Couldn't load test set"),
    //     config,
    //     "/tmp/text-generation",
    // );

    let mut text_items = Vec::new();
    text_items.push(
        "0, 5, 300, 100, 305, 217, \n1, 5, 200, 300, 50, 70, \n2, 5, 100, 100, 304, 116, "
            .to_string(),
    );
    //     text_items.push(
    //         "0, 5, 361, 161, 305, 217,
    // 1, 5, 232, 332, 50, 70,
    // 2, 5, 149, 149, 304, 116,
    // 0, 0, 361, 161, 330, -13,
    // 0, 2.5, 361, 161, 309, 90,
    // 0, 5, 361, 161, 305, 217,
    // 0, 15, 361, 161, 305, 217,
    // 0, 17.5, 361, 161, 312, 83,
    // 0, 20, 361, 161, 298, -22,
    // 1, 0, 232, 332, -17, 101,
    // 1, 2.5, 232, 332, 37, 86,
    // 1, 5, 232, 332, 50, 70,
    // 1, 15, 232, 332, 50, 70,
    // 1, 17.5, 232, 332, -5, 69,
    // 1, 20, 232, 332, -28, 106,
    // 2, 0, 149, 149, 305, -6,
    // 2, 2.5, 149, 149, 304, 57,
    // 2, 5, 149, 149, 304, 116,
    // 2, 15, 149, 149, 304, 116,
    // 2, 17.5, 149, 149, 306, 77,
    // 2, 20, 149, 149, 303, -11, "
    //             .to_string(),
    //     );
    //     text_items.push(
    //         "0, 5, 361, 161, 305, 217,
    // 1, 5, 232, 332, 50, 70,
    // 2, 5, 149, 149, 304, 116,
    // 0, 0, 361, 161, 0, 0,
    // 0, 2.5, 361, 161, 0, 0,
    // 0, 5, 361, 161, 0, 0,
    // 0, 15, 361, 161, 0, 0,
    // 0, 17.5, 361, 161, 0, 0,
    // 0, 20, 361, 161, 0, 0,
    // 1, 0, 232, 332, 0, 0,
    // 1, 2.5, 232, 332, 0, 0,
    // 1, 5, 232, 332, 0, 0,
    // 1, 15, 232, 332, 0, 0,
    // 1, 17.5, 232, 332, 0, 0,
    // 1, 20, 232, 332, 0, 0,
    // 2, 0, 149, 149, 0, 0,
    // 2, 2.5, 149, 149, 0, 0,
    // 2, 5, 149, 149, 0, 0,
    // 2, 15, 149, 149, 0, 0,
    // 2, 17.5, 149, 149, 0, 0,
    // 2, 20, 149, 149, 0, 0, "
    //             .to_string(),
    //     );
    // text_items.push("1 5 200 300 50 70 ".to_string());
    // text_items.push("2 5 100 100 304 116 ".to_string());

    let device = burn::tensor::Device::<Backend>::DiscreteGpu(0);

    infer_from_text::<Backend>(
        "/tmp/text-generation-e16-full-all-spaces",
        &device,
        text_items,
        512,
        1.0, // 1.0 should have no effect
    );
}
