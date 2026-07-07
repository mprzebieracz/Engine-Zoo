use super::*;
use crate::alphazero::network::{AlphaZeroNet, NetConfig};
use std::fs;
use std::sync::mpsc::sync_channel;
use tch::{nn, Device, Tensor};

fn tiny_cfg() -> NetConfig {
    NetConfig {
        input_channels: 1,
        height: 1,
        width: 1,
        num_res_blocks: 0,
        num_filters: 4,
        action_size: 5,
    }
}

fn batch(states: &[f32], legal_per_state: &[&[u32]]) -> EvalBatch {
    let mut legal = Vec::new();
    let mut offsets = Vec::with_capacity(legal_per_state.len() + 1);
    offsets.push(0);
    for actions in legal_per_state {
        legal.extend_from_slice(actions);
        offsets.push(legal.len() as u32);
    }

    EvalBatch {
        states: states.to_vec(),
        legal,
        offsets,
    }
}

fn shared(wait_for_count: usize) -> Arc<Shared> {
    Arc::new(Shared {
        pending: Mutex::new(Pending {
            tasks: Vec::new(),
            count: 0,
            stop: false,
        }),
        cv: Condvar::new(),
        wait_for_count,
        timeout: Duration::from_millis(10),
    })
}

fn direct_eval(net: &AlphaZeroNet, cfg: &NetConfig, batch: &EvalBatch) -> Vec<Evaluation> {
    let n = batch.len();
    let states = Tensor::from_slice(&batch.states).view([
        n as i64,
        cfg.input_channels,
        cfg.height,
        cfg.width,
    ]);
    let (policy, value) = net.forward_t(&states, false);
    let policy_flat: Vec<f32> = policy.contiguous().view(-1).try_into().unwrap();
    let values: Vec<f32> = value.contiguous().view(-1).try_into().unwrap();
    let action_size = cfg.action_size as usize;

    (0..n)
        .map(|row| {
            let begin = batch.offsets[row] as usize;
            let end = batch.offsets[row + 1] as usize;
            let logits = batch.legal[begin..end]
                .iter()
                .map(|&a| policy_flat[row * action_size + a as usize])
                .collect();
            Evaluation {
                logits,
                value: values[row],
            }
        })
        .collect()
}

fn assert_close(actual: &[Evaluation], expected: &[Evaluation]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.logits.len(), expected.logits.len());
        for (&a, &e) in actual.logits.iter().zip(&expected.logits) {
            assert!((a - e).abs() < 1e-5, "logit {a} != {e}");
        }
        assert!(
            (actual.value - expected.value).abs() < 1e-5,
            "value {} != {}",
            actual.value,
            expected.value
        );
    }
}

fn save_weights(cfg: &NetConfig) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "engine-zoo-batcher-test-{}-{}.safetensors",
        std::process::id(),
        rand::random::<u64>()
    ));
    let vs = nn::VarStore::new(Device::Cpu);
    let _net = AlphaZeroNet::new(&vs.root(), cfg);
    vs.save(&path).unwrap();
    path
}

#[test]
fn worker_evaluate_returns_only_requested_legal_logits() {
    tch::manual_seed(7);
    let cfg = tiny_cfg();
    let vs = nn::VarStore::new(Device::Cpu);
    let net = AlphaZeroNet::new(&vs.root(), &cfg);
    let request = batch(&[0.25, -0.5], &[&[4, 1], &[0, 3, 2]]);
    let expected = direct_eval(&net, &cfg, &request);

    let mut worker = Worker::new(net, cfg, Device::Cpu, shared(4));
    let task = Task {
        batch: request,
        tx: sync_channel(1).0,
    };
    let actual = worker.evaluate(&[task], 2);

    assert_close(&actual, &expected);
}

#[test]
fn process_splits_combined_results_back_to_each_task() {
    tch::manual_seed(11);
    let cfg = tiny_cfg();
    let vs = nn::VarStore::new(Device::Cpu);
    let net = AlphaZeroNet::new(&vs.root(), &cfg);
    let first = batch(&[1.0], &[&[2, 0]]);
    let second = batch(&[-1.0, 0.5], &[&[4], &[1, 3, 0]]);

    let mut combined = first.states.clone();
    combined.extend_from_slice(&second.states);
    let expected_first = direct_eval(&net, &cfg, &first);
    let expected_second = direct_eval(&net, &cfg, &second);

    let (tx1, rx1) = sync_channel(1);
    let (tx2, rx2) = sync_channel(1);
    let mut worker = Worker::new(net, cfg, Device::Cpu, shared(8));
    worker.process(vec![
        Task {
            batch: first,
            tx: tx1,
        },
        Task {
            batch: second,
            tx: tx2,
        },
    ]);

    assert_eq!(combined.len(), 3);
    assert_close(&rx1.recv().unwrap(), &expected_first);
    assert_close(&rx2.recv().unwrap(), &expected_second);
}

#[test]
fn client_evaluate_round_trips_through_worker_thread() {
    tch::manual_seed(19);
    let cfg = tiny_cfg();
    let weights = save_weights(&cfg);
    let mut direct_vs = nn::VarStore::new(Device::Cpu);
    let direct_net = AlphaZeroNet::new(&direct_vs.root(), &cfg);
    direct_vs.load(&weights).unwrap();

    let request = batch(&[0.0, 2.0], &[&[1, 4, 0], &[3]]);
    let expected = direct_eval(&direct_net, &cfg, &request);

    let batcher = Batcher::new(&cfg, &weights, Device::Cpu, 2, Duration::from_millis(50)).unwrap();
    let mut client = batcher.client();
    let actual = client.evaluate(&request);

    assert_close(&actual, &expected);
    fs::remove_file(weights).unwrap();
}

#[test]
fn staging_grows_without_shrinking_width() {
    let mut slot = None;
    let first = Worker::staging(&mut slot, 2, 5, Kind::Float, Device::Cpu);
    assert_eq!(first.size(), [2, 5]);

    let second = Worker::staging(&mut slot, 4, 3, Kind::Float, Device::Cpu);
    assert_eq!(second.size(), [4, 5]);
}
