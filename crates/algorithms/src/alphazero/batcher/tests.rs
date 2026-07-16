use super::*;
use crate::alphazero::network::{AlphaZeroNet, NetConfig, Network, NetworkConfig};
use crate::alphazero::representation::Action;
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

fn batch(states: &[f32], legal_per_state: &[&[u32]]) -> EncodedEvalBatch {
    let mut legal_actions = Vec::new();
    let mut offsets = Vec::with_capacity(legal_per_state.len() + 1);
    offsets.push(0);
    for actions in legal_per_state {
        legal_actions.extend(actions.iter().copied().map(Action::new));
        offsets.push(legal_actions.len() as u32);
    }

    EncodedEvalBatch {
        states: states.to_vec(),
        legal_actions,
        offsets,
    }
}

fn shared(wait_for_count: usize) -> Arc<Shared> {
    Arc::new(Shared {
        pending: Mutex::new(Pending::new()),
        cv: Condvar::new(),
        wait_for_count,
        timeout: Duration::from_millis(10),
    })
}

fn direct_eval(net: &AlphaZeroNet, cfg: &NetConfig, batch: &EncodedEvalBatch) -> Vec<Evaluation> {
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
            let logits = batch.legal_actions[begin..end]
                .iter()
                .map(|&a| policy_flat[row * action_size + a.index()])
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

fn save_seeded_weights(cfg: &NetConfig, seed: i64) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "engine-zoo-batcher-test-{}-{}-{}.safetensors",
        std::process::id(),
        seed,
        rand::random::<u64>()
    ));
    tch::manual_seed(seed);
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

    let mut worker = Worker::new(
        vs,
        Network::Legacy(net),
        NetworkConfig::Legacy(cfg),
        Device::Cpu,
        shared(4),
        InferencePrecision::Fp32,
    );
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
    let mut worker = Worker::new(
        vs,
        Network::Legacy(net),
        NetworkConfig::Legacy(cfg),
        Device::Cpu,
        shared(8),
        InferencePrecision::Fp32,
    );
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
    assert_close(&rx1.recv().unwrap().evaluations, &expected_first);
    assert_close(&rx2.recv().unwrap().evaluations, &expected_second);
}

#[test]
fn client_evaluate_round_trips_through_worker_thread() {
    tch::manual_seed(19);
    let cfg = tiny_cfg();
    let weights = save_weights(&cfg);
    let mut direct_vs = nn::VarStore::new(Device::Cpu);
    let direct_net = AlphaZeroNet::new(&direct_vs.root(), &cfg);
    direct_vs.load(&weights).unwrap();

    let mut request = batch(&[0.0, 2.0], &[&[1, 4, 0], &[3]]);
    let expected = direct_eval(&direct_net, &cfg, &request);

    let batcher = Batcher::new(&cfg, &weights, Device::Cpu, 2, Duration::from_millis(50)).unwrap();
    let mut client = batcher.client();
    let actual = client.evaluate(&mut request);

    assert_close(&actual, &expected);
    fs::remove_file(weights).unwrap();
}

#[test]
fn client_returns_the_submitted_vector_allocations_for_reuse() {
    let cfg = tiny_cfg();
    let weights = save_weights(&cfg);
    let batcher = Batcher::new(&cfg, &weights, Device::Cpu, 1, Duration::from_millis(1)).unwrap();
    let mut client = batcher.client();
    let mut request = batch(&[0.25, 0.5], &[&[0, 2], &[1, 3, 4]]);
    request.states.reserve(32);
    request.legal_actions.reserve(32);
    request.offsets.reserve(32);

    let pointers = (
        request.states.as_ptr(),
        request.legal_actions.as_ptr(),
        request.offsets.as_ptr(),
    );
    let capacities = (
        request.states.capacity(),
        request.legal_actions.capacity(),
        request.offsets.capacity(),
    );

    let evaluations = client.evaluate(&mut request);

    assert_eq!(evaluations.len(), 2);
    assert_eq!(request.states.as_ptr(), pointers.0);
    assert_eq!(request.legal_actions.as_ptr(), pointers.1);
    assert_eq!(request.offsets.as_ptr(), pointers.2);
    assert_eq!(request.states.capacity(), capacities.0);
    assert_eq!(request.legal_actions.capacity(), capacities.1);
    assert_eq!(request.offsets.capacity(), capacities.2);
    fs::remove_file(weights).unwrap();
}

#[test]
fn batcher_stats_record_submission_and_coalescing_sizes() {
    let cfg = tiny_cfg();
    let weights = save_weights(&cfg);
    let batcher = Batcher::new(&cfg, &weights, Device::Cpu, 1, Duration::from_millis(1)).unwrap();
    let before = batcher.stats();
    let mut client = batcher.client();
    let mut request = batch(&[0.25, 0.5], &[&[0], &[1]]);

    let _ = client.evaluate(&mut request);
    let stats = batcher.stats().since(before);

    assert_eq!(stats.submitted_batches, 1);
    assert_eq!(stats.submitted_states, 2);
    assert_eq!(stats.inference_batches, 1);
    assert_eq!(stats.coalesced_extra_requests, 0);
    assert!(stats.max_submitted_batch >= 2);
    assert!(stats.max_inference_batch >= 2);
    fs::remove_file(weights).unwrap();
}

#[test]
fn batcher_reloads_weights_without_restarting_client() {
    let cfg = tiny_cfg();
    let first_weights = save_seeded_weights(&cfg, 101);
    let second_weights = save_seeded_weights(&cfg, 202);

    let mut request = batch(&[0.75, -1.25], &[&[0, 2, 4], &[1, 3]]);

    let mut first_vs = nn::VarStore::new(Device::Cpu);
    let first_net = AlphaZeroNet::new(&first_vs.root(), &cfg);
    first_vs.load(&first_weights).unwrap();
    let expected_first = direct_eval(&first_net, &cfg, &request);

    let mut second_vs = nn::VarStore::new(Device::Cpu);
    let second_net = AlphaZeroNet::new(&second_vs.root(), &cfg);
    second_vs.load(&second_weights).unwrap();
    let expected_second = direct_eval(&second_net, &cfg, &request);

    let batcher = Batcher::new(
        &cfg,
        &first_weights,
        Device::Cpu,
        2,
        Duration::from_millis(50),
    )
    .unwrap();
    let mut client = batcher.client();
    assert_close(&client.evaluate(&mut request), &expected_first);

    batcher.reload_weights(&second_weights).unwrap();
    assert_close(&client.evaluate(&mut request), &expected_second);

    fs::remove_file(first_weights).unwrap();
    fs::remove_file(second_weights).unwrap();
}

#[test]
fn fp16_inference_is_rejected_on_cpu() {
    let cfg = tiny_cfg();
    let weights = save_weights(&cfg);
    let result = Batcher::new_with_precision(
        &cfg,
        &weights,
        Device::Cpu,
        1,
        Duration::from_millis(1),
        InferencePrecision::Fp16,
    );
    let err = match result {
        Ok(_) => panic!("FP16 inference on CPU should fail"),
        Err(err) => err,
    };

    assert!(err
        .to_string()
        .contains("FP16 inference is only supported on CUDA"));
    fs::remove_file(weights).unwrap();
}

#[test]
fn fp16_cuda_legal_gather_matches_fp32() {
    if !tch::Cuda::is_available() {
        return;
    }
    let cfg = tiny_cfg();
    let weights = save_seeded_weights(&cfg, 303);
    let device = Device::Cuda(0);
    let fp32 = Batcher::new_with_precision(
        &cfg,
        &weights,
        device,
        2,
        Duration::from_millis(1),
        InferencePrecision::Fp32,
    )
    .unwrap();
    let fp16 = Batcher::new_with_precision(
        &cfg,
        &weights,
        device,
        2,
        Duration::from_millis(1),
        InferencePrecision::Fp16,
    )
    .unwrap();
    let mut fp32_client = fp32.client();
    let mut fp16_client = fp16.client();
    let mut request32 = batch(&[0.125, -0.75], &[&[4, 0, 2], &[1, 3]]);
    let mut request16 = batch(&[0.125, -0.75], &[&[4, 0, 2], &[1, 3]]);

    let actual32 = fp32_client.evaluate(&mut request32);
    let actual16 = fp16_client.evaluate(&mut request16);
    assert_eq!(actual32.len(), actual16.len());
    for (fp32, fp16) in actual32.iter().zip(&actual16) {
        for (&a, &b) in fp32.logits.iter().zip(&fp16.logits) {
            assert!((a - b).abs() < 5e-3, "FP32 logit {a} != FP16 logit {b}");
        }
        assert!((fp32.value - fp16.value).abs() < 5e-3);
    }
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
