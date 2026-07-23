use super::*;
use crate::representation::Action;
use search::PositionValue;
use std::path::Path;
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone)]
struct MockBackend {
    calls: Arc<Mutex<Vec<usize>>>,
    fail: bool,
    reloads: Arc<Mutex<usize>>,
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl MockBackend {
    fn new() -> (Self, Arc<Mutex<Vec<usize>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                calls: Arc::clone(&calls),
                fail: false,
                reloads: Arc::new(Mutex::new(0)),
                events: Arc::new(Mutex::new(Vec::new())),
            },
            calls,
        )
    }
}

impl InferenceBackend for MockBackend {
    fn evaluate(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        self.calls.lock().unwrap().push(batch.len());
        self.events.lock().unwrap().push("evaluate");
        anyhow::ensure!(!self.fail, "mock backend failure");
        Ok((0..batch.len())
            .map(|row| {
                let begin = batch.offsets[row] as usize;
                let end = batch.offsets[row + 1] as usize;
                Evaluation {
                    logits: batch.legal_actions[begin..end]
                        .iter()
                        .map(|action| action.as_u32() as f32)
                        .collect(),
                    value: PositionValue::new(batch.states[row]).unwrap(),
                }
            })
            .collect())
    }

    fn reload_weights(&mut self, _path: &Path) -> Result<()> {
        *self.reloads.lock().unwrap() += 1;
        self.events.lock().unwrap().push("reload");
        Ok(())
    }
}

fn config(preferred: usize, max: usize) -> BatcherConfig {
    BatcherConfig {
        preferred_batch_size: preferred,
        max_batch_size: max,
        max_wait: Duration::from_millis(100),
        max_queued_states: 32,
    }
}

fn batch(states: &[f32], legal: &[&[u32]]) -> EncodedEvalBatch {
    let mut legal_actions = Vec::new();
    let mut offsets = vec![0];
    for actions in legal {
        legal_actions.extend(actions.iter().copied().map(Action::new));
        offsets.push(legal_actions.len() as u32);
    }
    EncodedEvalBatch {
        states: states.to_vec(),
        legal_actions,
        offsets,
    }
}

#[test]
fn config_rejects_invalid_limits() {
    assert!(config(0, 1).validate().is_err());
    assert!(config(2, 1).validate().is_err());
    let mut invalid = config(1, 1);
    invalid.max_queued_states = 0;
    assert!(invalid.validate().is_err());
}

#[test]
fn one_request_round_trips_and_returns_allocations() {
    let (backend, _) = MockBackend::new();
    let batcher = Batcher::with_backend(backend, config(1, 4)).unwrap();
    let mut client = batcher.client();
    let mut request = batch(&[0.25, -0.5], &[&[3, 1], &[4]]);
    request.states.reserve(16);
    let pointer = request.states.as_ptr();
    let result = client.evaluate(&mut request).unwrap();
    assert_eq!(
        result
            .iter()
            .map(|item| item.logits.clone())
            .collect::<Vec<_>>(),
        vec![vec![3.0, 1.0], vec![4.0]]
    );
    assert_eq!(request.states.as_ptr(), pointer);
}

#[test]
fn simultaneous_requests_coalesce() {
    let (backend, calls) = MockBackend::new();
    let batcher = Arc::new(Batcher::with_backend(backend, config(2, 4)).unwrap());
    let start = Arc::new(Barrier::new(3));
    let mut handles = Vec::new();
    for state in [0.0, 1.0] {
        let batcher = Arc::clone(&batcher);
        let start = Arc::clone(&start);
        handles.push(thread::spawn(move || {
            let mut request = batch(&[state], &[&[1]]);
            start.wait();
            batcher.client().evaluate(&mut request).unwrap()
        }));
    }
    start.wait();
    for handle in handles {
        assert_eq!(handle.join().unwrap().len(), 1);
    }
    assert_eq!(*calls.lock().unwrap(), vec![2]);
}

#[test]
fn timeout_flushes_partial_batch() {
    let (backend, calls) = MockBackend::new();
    let mut cfg = config(4, 4);
    cfg.max_wait = Duration::from_millis(5);
    let batcher = Batcher::with_backend(backend, cfg).unwrap();
    let mut request = batch(&[1.0], &[&[0]]);
    assert_eq!(batcher.client().evaluate(&mut request).unwrap().len(), 1);
    assert_eq!(*calls.lock().unwrap(), vec![1]);
    assert_eq!(batcher.stats().partial_batches, 1);
}

#[test]
fn max_batch_splits_and_reassembles_one_large_request_in_order() {
    let (backend, calls) = MockBackend::new();
    let batcher = Batcher::with_backend(backend, config(1, 2)).unwrap();
    let mut request = batch(
        &[0.0, 0.25, 0.5, 0.75, 1.0],
        &[&[0], &[1], &[2], &[3], &[4]],
    );
    let evaluations = batcher.client().evaluate(&mut request).unwrap();
    assert_eq!(*calls.lock().unwrap(), vec![2, 2, 1]);
    assert_eq!(
        evaluations
            .iter()
            .map(|item| item.logits[0])
            .collect::<Vec<_>>(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(batcher.stats().lifetime_max_inference_batch, 2);
    assert_eq!(batcher.stats().split_batches, 2);
}

#[test]
fn backend_failure_reaches_current_and_future_clients() {
    let (mut backend, _) = MockBackend::new();
    backend.fail = true;
    let batcher = Batcher::with_backend(backend, config(1, 2)).unwrap();
    let mut first = batch(&[0.0], &[&[0]]);
    assert!(matches!(
        batcher.client().evaluate(&mut first),
        Err(BatcherError::Backend(_))
    ));
    let mut next = batch(&[1.0], &[&[0]]);
    assert!(matches!(
        batcher.client().evaluate(&mut next),
        Err(BatcherError::Backend(_))
    ));
}

#[test]
fn reload_is_a_barrier_before_the_next_inference_pass() {
    let (backend, _) = MockBackend::new();
    let events = Arc::clone(&backend.events);
    let mut cfg = config(2, 2);
    cfg.max_wait = Duration::from_millis(50);
    let batcher = Arc::new(Batcher::with_backend(backend, cfg).unwrap());
    let evaluation_batcher = Arc::clone(&batcher);
    let evaluation = thread::spawn(move || {
        let mut request = batch(&[0.0], &[&[0]]);
        evaluation_batcher.client().evaluate(&mut request)
    });
    thread::sleep(Duration::from_millis(5));
    batcher.reload_weights(Path::new("unused")).unwrap();
    assert_eq!(batcher.client().cache_namespace(), 1);
    evaluation.join().unwrap().unwrap();
    assert_eq!(*events.lock().unwrap(), vec!["evaluate", "reload"]);
}

#[test]
fn dropping_batcher_unblocks_waiting_client() {
    let (backend, _) = MockBackend::new();
    let batcher = Arc::new(Batcher::with_backend(backend, config(2, 2)).unwrap());
    let mut client = batcher.client();
    let waiter = thread::spawn(move || {
        let mut request = batch(&[0.0], &[&[0]]);
        client.evaluate(&mut request)
    });
    thread::sleep(Duration::from_millis(5));
    drop(batcher);
    assert!(matches!(
        waiter.join().unwrap(),
        Err(BatcherError::Shutdown)
    ));
}
