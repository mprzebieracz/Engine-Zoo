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
