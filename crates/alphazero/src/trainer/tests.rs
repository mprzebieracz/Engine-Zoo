use super::*;

fn tiny_net_config() -> NetConfig {
    NetConfig {
        input_channels: 1,
        height: 1,
        width: 1,
        num_res_blocks: 0,
        num_filters: 2,
        action_size: 2,
    }
}

#[test]
fn empty_replay_returns_none() {
    let cfg = TrainConfig {
        micro_batch_size: 1,
        batch_size: 1,
        train_steps: 1,
        progress_every: 0,
        ..TrainConfig::default()
    };
    let net_cfg = tiny_net_config();
    let vs = nn::VarStore::new(Device::Cpu);
    let net = AlphaZeroNet::new(&vs.root(), &net_cfg);
    let mut optimizer = build_optimizer(&vs, &cfg).unwrap();
    let replay = ReplayBuffer::new(1, net_cfg.state_size(), net_cfg.action_size as usize);

    assert!(train(&net, &mut optimizer, &replay, Device::Cpu, &net_cfg, &cfg).is_none());
}

#[test]
fn rejects_invalid_training_config() {
    let invalid = [
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
    ];

    for cfg in invalid {
        assert!(cfg.validate().is_err());
    }
}

#[test]
fn sparse_policy_loss_matches_dense_reference_and_uneven_chunks() {
    let logits = Tensor::from_slice(&[
        0.2f32, -0.7, 1.1, 0.4, -0.2, // row 0
        -1.0, 0.3, 0.8, -0.4, 0.6, // row 1
        0.9, -0.1, 0.0, 1.2, -0.8, // row 2
    ])
    .view([3, 5]);
    let dense = Tensor::from_slice(&[
        0.0f32, 0.25, 0.0, 0.75, 0.0, // row 0
        0.4, 0.0, 0.6, 0.0, 0.0, // row 1
        0.0, 0.0, 0.1, 0.3, 0.6, // row 2
    ])
    .view([3, 5]);
    let actions = Tensor::from_slice(&[1i64, 3, 0, 2, 2, 3, 4]);
    let probabilities = Tensor::from_slice(&[0.25f32, 0.75, 0.4, 0.6, 0.1, 0.3, 0.6]);
    let rows = Tensor::from_slice(&[0i64, 0, 1, 1, 2, 2, 2]);

    let dense_loss = dense_policy_cross_entropy(&logits, &dense);
    let sparse_loss = sparse_policy_cross_entropy(&logits, &actions, &probabilities, &rows, 3);
    assert!((dense_loss.double_value(&[]) - sparse_loss.double_value(&[])).abs() < 1e-6);

    // Simulate micro_batch_size=2: the final chunk has one position and must
    // retain exactly its sample weight in the optimizer-step loss.
    let first = sparse_policy_cross_entropy(
        &logits.narrow(0, 0, 2),
        &actions.narrow(0, 0, 4),
        &probabilities.narrow(0, 0, 4),
        &rows.narrow(0, 0, 4),
        2,
    );
    let last = sparse_policy_cross_entropy(
        &logits.narrow(0, 2, 1),
        &actions.narrow(0, 4, 3),
        &probabilities.narrow(0, 4, 3),
        &(rows.narrow(0, 4, 3) - 2),
        1,
    );
    let chunked = first * (2.0 / 3.0) + last * (1.0 / 3.0);
    assert!((dense_loss.double_value(&[]) - chunked.double_value(&[])).abs() < 1e-6);
}

#[test]
fn sparse_policy_loss_handles_positions_without_policy_mass() {
    let logits = Tensor::from_slice(&[0.2f32, -0.4, 0.8, 1.0, 0.0, -1.0]).view([2, 3]);
    let dense = Tensor::zeros([2, 3], (Kind::Float, Device::Cpu));
    let actions = Tensor::from_slice::<i64>(&[]);
    let probabilities = Tensor::from_slice::<f32>(&[]);
    let rows = Tensor::from_slice::<i64>(&[]);

    let dense_loss = dense_policy_cross_entropy(&logits, &dense);
    let sparse_loss = sparse_policy_cross_entropy(&logits, &actions, &probabilities, &rows, 2);
    assert!(sparse_loss.double_value(&[]).is_finite());
    assert!((dense_loss.double_value(&[]) - sparse_loss.double_value(&[])).abs() < 1e-7);
}
