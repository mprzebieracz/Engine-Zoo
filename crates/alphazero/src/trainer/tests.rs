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
            prefetch_depth: 0,
            ..TrainConfig::default()
        },
        TrainConfig {
            prefetch_depth: 3,
            ..TrainConfig::default()
        },
    ] {
        assert!(cfg.validate().is_err());
    }
}

#[test]
fn zero_train_steps_skips_training() {
    use crate::representation::Connect4AzRepresentation;
    use crate::{ModelSpec, Network, ReplayBuffer};
    use games::Connect4;
    use tch::{nn, Device};

    let vs = nn::VarStore::new(Device::Cpu);
    let network = Network::new(&vs.root(), &ModelSpec::connect4_basic(1, 4)).unwrap();
    let mut optimizer = build_optimizer(&vs, &TrainConfig::default()).unwrap();
    let replay = ReplayBuffer::<Connect4>::new(8, 7);
    let config = TrainConfig {
        train_steps: 0,
        ..TrainConfig::default()
    };

    assert!(train(
        &network,
        &mut optimizer,
        &replay,
        &Connect4AzRepresentation,
        Device::Cpu,
        &config,
        7,
        0,
    )
    .is_none());
}

#[test]
fn empty_replay_stops_the_prefetcher_without_waiting_for_train_steps() {
    use crate::representation::Connect4AzRepresentation;
    use crate::{ModelSpec, Network, ReplayBuffer};
    use games::Connect4;
    use tch::{nn, Device};

    let vs = nn::VarStore::new(Device::Cpu);
    let network = Network::new(&vs.root(), &ModelSpec::connect4_basic(1, 4)).unwrap();
    let mut optimizer = build_optimizer(&vs, &TrainConfig::default()).unwrap();
    let replay = ReplayBuffer::<Connect4>::new(8, 7);
    let metrics = train(
        &network,
        &mut optimizer,
        &replay,
        &Connect4AzRepresentation,
        Device::Cpu,
        &TrainConfig {
            train_steps: 3,
            ..TrainConfig::default()
        },
        7,
        0,
    );
    assert!(metrics.is_none());
}

#[test]
fn prefetcher_closes_after_the_last_nonempty_batch() {
    use crate::representation::Connect4AzRepresentation;
    use crate::{Action, ModelSpec, Outcome, ReplaySample, SampleMetadata, TrainingWeights};
    use games::Connect4;
    use tch::{nn, Device};

    let vs = nn::VarStore::new(Device::Cpu);
    let network = Network::new(&vs.root(), &ModelSpec::connect4_basic(1, 4)).unwrap();
    let config = TrainConfig {
        batch_size: 1,
        train_steps: 2,
        ..Default::default()
    };
    let mut optimizer = build_optimizer(&vs, &config).unwrap();
    let replay = ReplayBuffer::<Connect4>::new(8, 7);
    replay.add([ReplaySample {
        state: Connect4::default(),
        policy: vec![(Action::new(0), 1.0)].into(),
        outcome: Outcome::Draw,
        weights: TrainingWeights::default(),
        metadata: SampleMetadata::default(),
    }]);

    let metrics = train(
        &network,
        &mut optimizer,
        &replay,
        &Connect4AzRepresentation,
        Device::Cpu,
        &config,
        7,
        0,
    )
    .unwrap();
    assert_eq!(metrics.train_steps, 2);
}

#[test]
fn replay_step_seeds_are_stable_and_step_specific() {
    assert_eq!(replay_seed(7, 11), replay_seed(7, 11));
    assert_ne!(replay_seed(7, 11), replay_seed(7, 12));
}

#[test]
fn learning_rate_schedules_are_explicit_and_validated() {
    let mut config = TrainConfig {
        lr: 1.0,
        learning_rate_schedule: LearningRateSchedule::LinearWarmupThenCosine {
            warmup_steps: 2,
            total_steps: 6,
            minimum_lr: 0.1,
        },
        ..Default::default()
    };
    assert_eq!(config.learning_rate_at(0), 0.5);
    assert_eq!(config.learning_rate_at(1), 1.0);
    assert!((config.learning_rate_at(6) - 0.1).abs() < f64::EPSILON);

    config.learning_rate_schedule = LearningRateSchedule::Piecewise {
        boundaries: vec![(2, 0.5), (5, 0.1)],
    };
    assert_eq!(config.learning_rate_at(0), 1.0);
    assert_eq!(config.learning_rate_at(2), 0.5);
    assert_eq!(config.learning_rate_at(5), 0.1);
    config.validate().unwrap();
}

#[test]
fn device_batch_rebases_sparse_rows_for_a_microbatch() {
    let batch = super::super::replay::ReplayBatch {
        states: Tensor::zeros([2, 1], (Kind::Float, Device::Cpu)),
        policies: super::super::replay::SparsePolicyBatch {
            actions: Tensor::from_slice(&[1i64, 2, 3]),
            probabilities: Tensor::from_slice(&[0.4f32, 0.6, 1.0]),
            rows: Tensor::from_slice(&[0i64, 0, 1]),
            offsets: vec![0, 2, 3],
        },
        outcomes: Tensor::from_slice(&[1i64, 2]),
        policy_weights: Tensor::ones([2], (Kind::Float, Device::Cpu)),
        policy_weight_sum: 2.0,
        value_weights: Tensor::ones([2], (Kind::Float, Device::Cpu)),
        value_weight_sum: 2.0,
    };
    let device = DeviceReplayBatch::from_cpu(batch, Device::Cpu);
    let policies = device.policy_rows(1, 1);
    assert_eq!(Vec::<i64>::try_from(&policies.actions).unwrap(), [3]);
    assert_eq!(Vec::<i64>::try_from(&policies.rows).unwrap(), [0]);
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
