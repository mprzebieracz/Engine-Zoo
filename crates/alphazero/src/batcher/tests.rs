use super::*;
use crate::representation::Action;
use search::PositionValue;
use std::path::Path;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::Duration;

enum FakeBehavior {
    Success,
    Error,
    WrongCardinality,
    BlockUntilShutdown {
        entered: SyncSender<()>,
        release: Receiver<()>,
    },
    WorkerStop,
    FailOnCall(usize),
}

struct MockBackend {
    calls: Arc<Mutex<Vec<usize>>>,
    behavior: FakeBehavior,
    reloads: Arc<Mutex<usize>>,
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl MockBackend {
    fn new() -> (Self, Arc<Mutex<Vec<usize>>>) {
        let calls = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                calls: Arc::clone(&calls),
                behavior: FakeBehavior::Success,
                reloads: Arc::new(Mutex::new(0)),
                events: Arc::new(Mutex::new(Vec::new())),
            },
            calls,
        )
    }
}

impl InferenceBackend for MockBackend {
    fn evaluate(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push(batch.len());
            calls.len()
        };
        self.events.lock().unwrap().push("evaluate");

        match &self.behavior {
            FakeBehavior::Success => {}
            FakeBehavior::Error => anyhow::bail!("mock backend failure"),
            FakeBehavior::WrongCardinality => return Ok(Vec::new()),
            FakeBehavior::BlockUntilShutdown { entered, release } => {
                entered.send(()).unwrap();
                release.recv().unwrap();
            }
            FakeBehavior::WorkerStop => panic!("simulated worker stop"),
            FakeBehavior::FailOnCall(failing_call) if call == *failing_call => {
                anyhow::bail!("mock backend failure on call {call}")
            }
            FakeBehavior::FailOnCall(_) => {}
        }

        Ok(evaluations(batch))
    }

    fn reload_weights(&mut self, _path: &Path) -> Result<()> {
        *self.reloads.lock().unwrap() += 1;
        self.events.lock().unwrap().push("reload");
        Ok(())
    }
}

fn evaluations(batch: &CombinedEncodedBatch) -> Vec<Evaluation> {
    (0..batch.len())
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
        .collect()
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BatchStorage {
    pointers: [usize; 3],
    capacities: [usize; 3],
}

impl BatchStorage {
    fn of(batch: &EncodedEvalBatch) -> Self {
        Self {
            pointers: [
                batch.states.as_ptr() as usize,
                batch.legal_actions.as_ptr() as usize,
                batch.offsets.as_ptr() as usize,
            ],
            capacities: [
                batch.states.capacity(),
                batch.legal_actions.capacity(),
                batch.offsets.capacity(),
            ],
        }
    }
}

fn reusable_batch(states: &[f32], legal: &[&[u32]]) -> EncodedEvalBatch {
    let mut batch = batch(states, legal);
    batch.states.reserve(31);
    batch.legal_actions.reserve(29);
    batch.offsets.reserve(23);

    batch
}

fn assert_batch_restored(
    actual: &EncodedEvalBatch,
    expected: &EncodedEvalBatch,
    storage: BatchStorage,
) {
    assert_eq!(actual.states, expected.states);
    assert_eq!(actual.legal_actions, expected.legal_actions);
    assert_eq!(actual.offsets, expected.offsets);
    assert_eq!(BatchStorage::of(actual), storage);
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
fn evaluate_restores_batch_after_success() {
    let (backend, _) = MockBackend::new();
    let batcher = Batcher::with_backend(backend, config(1, 4)).unwrap();
    let mut client = batcher.client();
    let mut request = reusable_batch(&[0.25, -0.5], &[&[3, 1], &[4]]);
    let expected = batch(&[0.25, -0.5], &[&[3, 1], &[4]]);
    let storage = BatchStorage::of(&request);

    let result = client.evaluate(&mut request).unwrap();

    assert_eq!(
        result
            .iter()
            .map(|item| item.logits.clone())
            .collect::<Vec<_>>(),
        vec![vec![3.0, 1.0], vec![4.0]]
    );
    assert_batch_restored(&request, &expected, storage);
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
fn evaluate_restores_batch_after_backend_error() {
    let (mut backend, _) = MockBackend::new();
    backend.behavior = FakeBehavior::Error;
    let batcher = Batcher::with_backend(backend, config(1, 2)).unwrap();
    let mut first = reusable_batch(&[0.0], &[&[0]]);
    let expected = batch(&[0.0], &[&[0]]);
    let storage = BatchStorage::of(&first);

    assert!(matches!(
        batcher.client().evaluate(&mut first),
        Err(BatcherError::Backend(_))
    ));
    assert_batch_restored(&first, &expected, storage);

    let mut next = batch(&[1.0], &[&[0]]);
    assert!(matches!(
        batcher.client().evaluate(&mut next),
        Err(BatcherError::Backend(_))
    ));
}

#[test]
fn evaluate_restores_batch_after_wrong_cardinality() {
    let (mut backend, _) = MockBackend::new();
    backend.behavior = FakeBehavior::WrongCardinality;
    let batcher = Batcher::with_backend(backend, config(1, 2)).unwrap();
    let mut request = reusable_batch(&[0.0], &[&[0]]);
    let expected = batch(&[0.0], &[&[0]]);
    let storage = BatchStorage::of(&request);

    assert!(matches!(
        batcher.client().evaluate(&mut request),
        Err(BatcherError::Backend(_))
    ));
    assert_batch_restored(&request, &expected, storage);
}

#[test]
fn reload_is_a_barrier_before_the_next_inference_pass() {
    let (mut backend, _) = MockBackend::new();
    let events = Arc::clone(&backend.events);
    let (entered_tx, entered_rx) = sync_channel(1);
    let (release_tx, release_rx) = sync_channel(1);
    backend.behavior = FakeBehavior::BlockUntilShutdown {
        entered: entered_tx,
        release: release_rx,
    };
    let batcher = Arc::new(Batcher::with_backend(backend, config(1, 2)).unwrap());
    let evaluation_batcher = Arc::clone(&batcher);
    let evaluation = thread::spawn(move || {
        let mut request = batch(&[0.0], &[&[0]]);
        evaluation_batcher.client().evaluate(&mut request)
    });
    entered_rx.recv().unwrap();

    let reload_batcher = Arc::clone(&batcher);
    let reload = thread::spawn(move || reload_batcher.reload_weights(Path::new("unused")));
    let pending = batcher.shared.pending.lock().unwrap();
    let pending = batcher
        .shared
        .cv
        .wait_while(pending, |state| {
            !state
                .commands
                .iter()
                .any(|command| matches!(command, Command::Reload(_)))
        })
        .unwrap();
    drop(pending);
    release_tx.send(()).unwrap();

    reload.join().unwrap().unwrap();
    assert_eq!(batcher.client().cache_namespace(), 1);
    evaluation.join().unwrap().unwrap();
    assert_eq!(*events.lock().unwrap(), vec!["evaluate", "reload"]);
}

#[test]
fn evaluate_restores_batch_after_shutdown() {
    let (mut backend, _) = MockBackend::new();
    let (entered_tx, entered_rx) = sync_channel(1);
    let (release_tx, release_rx) = sync_channel(1);
    backend.behavior = FakeBehavior::BlockUntilShutdown {
        entered: entered_tx,
        release: release_rx,
    };
    let batcher = Batcher::with_backend(backend, config(1, 2)).unwrap();
    let shared = Arc::clone(&batcher.shared);
    let mut client = batcher.client();
    let waiter = thread::spawn(move || {
        let mut request = reusable_batch(&[0.0], &[&[0]]);
        let storage = BatchStorage::of(&request);
        let result = client.evaluate(&mut request);

        (result, request, storage)
    });
    entered_rx.recv().unwrap();

    let dropper = thread::spawn(move || drop(batcher));
    let pending = shared.pending.lock().unwrap();
    let pending = shared.cv.wait_while(pending, |state| !state.stop).unwrap();
    drop(pending);
    release_tx.send(()).unwrap();

    let (result, request, storage) = waiter.join().unwrap();
    dropper.join().unwrap();

    assert!(matches!(result, Err(BatcherError::Shutdown)));
    assert_batch_restored(&request, &batch(&[0.0], &[&[0]]), storage);
}

#[test]
fn evaluate_restores_batch_after_worker_stop() {
    let (mut backend, _) = MockBackend::new();
    backend.behavior = FakeBehavior::WorkerStop;
    let batcher = Batcher::with_backend(backend, config(1, 2)).unwrap();
    let mut request = reusable_batch(&[0.0], &[&[0]]);
    let storage = BatchStorage::of(&request);

    let result = batcher.client().evaluate(&mut request);

    assert!(matches!(result, Err(BatcherError::WorkerStopped)));
    assert_batch_restored(&request, &batch(&[0.0], &[&[0]]), storage);
}

#[test]
fn split_task_restores_batch_after_later_pass_failure() {
    let (mut backend, calls) = MockBackend::new();
    backend.behavior = FakeBehavior::FailOnCall(2);
    let batcher = Batcher::with_backend(backend, config(1, 2)).unwrap();
    let mut request = reusable_batch(&[0.0, 0.25, 0.5], &[&[0], &[1], &[2]]);
    let expected = batch(&[0.0, 0.25, 0.5], &[&[0], &[1], &[2]]);
    let storage = BatchStorage::of(&request);

    let result = batcher.client().evaluate(&mut request);

    assert!(matches!(result, Err(BatcherError::Backend(_))));
    assert_eq!(*calls.lock().unwrap(), vec![2, 1]);
    assert_batch_restored(&request, &expected, storage);
}
