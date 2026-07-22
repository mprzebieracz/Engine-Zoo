use super::*;

#[test]
fn rejects_invalid_training_config() {
    for cfg in [
        TrainConfig {
            micro_batch_size: 0,
            ..TrainConfig::default()
        },
        TrainConfig {
            batch_size: 0,
            ..TrainConfig::default()
        },
        TrainConfig {
            train_steps: 0,
            ..TrainConfig::default()
        },
    ] {
        assert!(cfg.validate().is_err());
    }
}

#[test]
fn wdl_uses_categorical_outcomes_and_weights() {
    let logits = Tensor::from_slice(&[2.0f32, 1.0, -1.0, -1.0, 1.0, 2.0]).view([2, 3]);
    let outcomes = Tensor::from_slice(&[0i64, 2]);
    let weights = Tensor::from_slice(&[1.0f32, 0.0]);
    let loss = wdl_cross_entropy(&logits, &outcomes, &weights);
    assert!(
        (loss.double_value(&[])
            - -(2.0f64.exp() / (2.0f64.exp() + 1.0f64.exp() + (-1.0f64).exp())).ln())
        .abs()
            < 1e-5
    );
}
