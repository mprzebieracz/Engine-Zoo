//! Dynamic request batching, independent from the inference implementation.
//!
//! The worker owns queueing, coalescing, splitting, and lifecycle handling.
//! A backend only sees one contiguous encoded batch at a time.

use super::evaluator::{EncodedEvalBatch, EncodedEvaluator, Evaluation};
use super::network::ModelSpec;
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tch::Device;

mod tch_backend;

pub use tch_backend::{InferencePrecision, TchInferenceBackend};

/// One contiguous input passed to an inference implementation.
#[derive(Default)]
pub struct CombinedEncodedBatch {
    pub states: Vec<f32>,
    pub legal_actions: Vec<super::representation::Action>,
    pub offsets: Vec<u32>,
}

impl CombinedEncodedBatch {
    fn new() -> Self {
        Self {
            states: Vec::new(),
            legal_actions: Vec::new(),
            offsets: vec![0],
        }
    }

    pub fn len(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn append_rows(
        &mut self,
        batch: &EncodedEvalBatch,
        rows: std::ops::Range<usize>,
    ) -> Result<()> {
        anyhow::ensure!(
            rows.start <= rows.end && rows.end <= batch.len(),
            "invalid batch row range"
        );
        let state_size = if batch.is_empty() {
            0
        }
        else {
            anyhow::ensure!(
                batch.states.len() % batch.len() == 0,
                "encoded state count does not match rows"
            );
            batch.states.len() / batch.len()
        };
        self.states
            .extend_from_slice(&batch.states[rows.start * state_size..rows.end * state_size]);
        for row in rows {
            let begin = batch.offsets[row] as usize;
            let end = batch.offsets[row + 1] as usize;
            anyhow::ensure!(
                begin <= end && end <= batch.legal_actions.len(),
                "invalid legal-action offsets"
            );
            self.legal_actions
                .extend_from_slice(&batch.legal_actions[begin..end]);
            self.offsets.push(self.legal_actions.len() as u32);
        }
        Ok(())
    }
}

/// The narrow boundary between dynamic batching and model execution.
pub trait InferenceBackend: Send + 'static {
    fn evaluate(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>>;
    fn reload_weights(&mut self, path: &Path) -> Result<()>;
}

/// Dynamic batching policy. A single request bigger than `max_queued_states`
/// is admitted only when no other states are queued, then split by
/// `max_batch_size`; this avoids making legitimate large requests impossible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatcherConfig {
    pub preferred_batch_size: usize,
    pub max_batch_size: usize,
    pub max_wait: Duration,
    pub max_queued_states: usize,
}

impl BatcherConfig {
    pub fn validate(self) -> Result<Self> {
        anyhow::ensure!(
            self.preferred_batch_size > 0,
            "preferred batch size must be at least one"
        );
        anyhow::ensure!(
            self.max_batch_size >= self.preferred_batch_size,
            "max batch size must be at least the preferred batch size"
        );
        anyhow::ensure!(
            self.max_queued_states > 0,
            "maximum queued states must be at least one"
        );
        Ok(self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BatcherError {
    Backend(String),
    Shutdown,
    WorkerStopped,
}

impl fmt::Display for BatcherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backend(message) => write!(f, "inference backend failed: {message}"),
            Self::Shutdown => f.write_str("batcher is shut down"),
            Self::WorkerStopped => f.write_str("batcher worker stopped"),
        }
    }
}

impl std::error::Error for BatcherError {}

type BatcherResult<T> = std::result::Result<T, BatcherError>;

/// Cross-thread dynamic batching for one shared inference backend.
pub struct Batcher {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

#[derive(Clone)]
pub struct BatcherClient {
    shared: Arc<Shared>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct BatcherStats {
    pub submitted_batches: u64,
    pub submitted_states: u64,
    pub inference_batches: u64,
    pub coalesced_extra_requests: u64,
    pub split_batches: u64,
    pub partial_batches: u64,
    pub total_states: u64,
    pub queue_wait_total: Duration,
    pub queue_wait_max: Duration,
    pub coalescing_wait_total: Duration,
    pub backend_execution_total: Duration,
    pub backend_execution_max: Duration,
    pub reload_count: u64,
    pub current_cache_namespace: u64,
    pub lifetime_max_submitted_batch: u64,
    pub lifetime_max_inference_batch: u64,
}

impl BatcherStats {
    pub fn since(self, earlier: Self) -> Self {
        Self {
            submitted_batches: self
                .submitted_batches
                .saturating_sub(earlier.submitted_batches),
            submitted_states: self
                .submitted_states
                .saturating_sub(earlier.submitted_states),
            inference_batches: self
                .inference_batches
                .saturating_sub(earlier.inference_batches),
            coalesced_extra_requests: self
                .coalesced_extra_requests
                .saturating_sub(earlier.coalesced_extra_requests),
            split_batches: self.split_batches.saturating_sub(earlier.split_batches),
            partial_batches: self.partial_batches.saturating_sub(earlier.partial_batches),
            total_states: self.total_states.saturating_sub(earlier.total_states),
            queue_wait_total: self
                .queue_wait_total
                .saturating_sub(earlier.queue_wait_total),
            queue_wait_max: self.queue_wait_max,
            coalescing_wait_total: self
                .coalescing_wait_total
                .saturating_sub(earlier.coalescing_wait_total),
            backend_execution_total: self
                .backend_execution_total
                .saturating_sub(earlier.backend_execution_total),
            backend_execution_max: self.backend_execution_max,
            reload_count: self.reload_count.saturating_sub(earlier.reload_count),
            current_cache_namespace: self.current_cache_namespace,
            lifetime_max_submitted_batch: self.lifetime_max_submitted_batch,
            lifetime_max_inference_batch: self.lifetime_max_inference_batch,
        }
    }
}

struct Task {
    batch: EncodedEvalBatch,
    next_row: usize,
    evaluations: Vec<Evaluation>,
    tx: SyncSender<BatcherResult<EvalResponse>>,
    submitted_at: Instant,
}

struct EvalResponse {
    batch: EncodedEvalBatch,
    evaluations: Vec<Evaluation>,
}

struct Reload {
    weights: PathBuf,
    tx: SyncSender<BatcherResult<()>>,
}

enum Command {
    Evaluate(Task),
    Reload(Reload),
}

struct Pending {
    commands: VecDeque<Command>,
    count: usize,
    stop: bool,
    terminal_error: Option<BatcherError>,
    stats: BatcherStats,
}

impl Pending {
    fn new() -> Self {
        Self {
            commands: VecDeque::new(),
            count: 0,
            stop: false,
            terminal_error: None,
            stats: BatcherStats::default(),
        }
    }
}

struct Shared {
    pending: Mutex<Pending>,
    cv: Condvar,
    config: BatcherConfig,
    evaluation_generation: AtomicU64,
}

enum Work {
    Evaluate,
    Reload(Reload),
    Stop,
}

struct WorkItem {
    task: Task,
    rows: std::ops::Range<usize>,
}

impl Batcher {
    pub fn with_backend<B: InferenceBackend>(backend: B, config: BatcherConfig) -> Result<Self> {
        let config = config.validate()?;
        let shared = Arc::new(Shared {
            pending: Mutex::new(Pending::new()),
            cv: Condvar::new(),
            config,
            evaluation_generation: AtomicU64::new(0),
        });
        let worker_shared = Arc::clone(&shared);
        let worker = std::thread::Builder::new()
            .name("batcher".into())
            .spawn(move || run_worker(Box::new(backend), worker_shared))
            .context("spawning batcher worker")?;
        Ok(Self {
            shared,
            worker: Some(worker),
        })
    }

    pub fn new_with_model(
        spec: ModelSpec,
        weights: &Path,
        device: Device,
        config: BatcherConfig,
    ) -> Result<Self> {
        Self::new_with_model_precision(spec, weights, device, config, InferencePrecision::Fp32)
    }

    pub fn new_with_model_precision(
        spec: ModelSpec,
        weights: &Path,
        device: Device,
        config: BatcherConfig,
        precision: InferencePrecision,
    ) -> Result<Self> {
        Self::with_backend(
            TchInferenceBackend::new(spec, weights, device, precision)?,
            config,
        )
    }

    pub fn client(&self) -> BatcherClient {
        BatcherClient {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn stats(&self) -> BatcherStats {
        self.shared.pending.lock().unwrap().stats
    }

    pub fn reload_weights(&self, weights: &Path) -> BatcherResult<()> {
        let (tx, rx) = sync_channel(1);
        let mut pending = self.shared.pending.lock().unwrap();
        if let Some(error) = &pending.terminal_error {
            return Err(error.clone());
        }
        if pending.stop {
            return Err(BatcherError::Shutdown);
        }
        pending.commands.push_back(Command::Reload(Reload {
            weights: weights.to_path_buf(),
            tx,
        }));
        drop(pending);
        self.shared.cv.notify_all();
        rx.recv().unwrap_or(Err(BatcherError::WorkerStopped))
    }
}

impl Drop for Batcher {
    fn drop(&mut self) {
        let mut pending = self.shared.pending.lock().unwrap();
        pending.stop = true;
        fail_pending(&mut pending, BatcherError::Shutdown);
        drop(pending);
        self.shared.cv.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl BatcherClient {
    pub fn cache_namespace(&self) -> u64 {
        self.shared.evaluation_generation.load(Ordering::Acquire)
    }

    pub fn evaluate(&mut self, batch: &mut EncodedEvalBatch) -> BatcherResult<Vec<Evaluation>> {
        let rows = batch.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        let (tx, rx) = sync_channel(1);
        let mut pending = self.shared.pending.lock().unwrap();
        while pending.terminal_error.is_none()
            && !pending.stop
            && pending.count != 0
            && pending.count.saturating_add(rows) > self.shared.config.max_queued_states
        {
            pending = self.shared.cv.wait(pending).unwrap();
        }
        if let Some(error) = &pending.terminal_error {
            return Err(error.clone());
        }
        if pending.stop {
            return Err(BatcherError::Shutdown);
        }
        let owned = std::mem::take(batch);
        pending.stats.submitted_batches += 1;
        pending.stats.submitted_states += rows as u64;
        pending.stats.total_states += rows as u64;
        pending.stats.lifetime_max_submitted_batch =
            pending.stats.lifetime_max_submitted_batch.max(rows as u64);
        pending.count += rows;
        pending.commands.push_back(Command::Evaluate(Task {
            batch: owned,
            next_row: 0,
            evaluations: Vec::with_capacity(rows),
            tx,
            submitted_at: Instant::now(),
        }));
        drop(pending);
        self.shared.cv.notify_all();
        match rx.recv().unwrap_or(Err(BatcherError::WorkerStopped)) {
            Ok(response) => {
                *batch = response.batch;
                Ok(response.evaluations)
            }
            Err(error) => Err(error),
        }
    }
}

impl EncodedEvaluator for BatcherClient {
    fn evaluate(
        &mut self,
        batch: &mut EncodedEvalBatch,
    ) -> Result<Vec<Evaluation>, search::EvaluationError> {
        BatcherClient::evaluate(self, batch)
            .map_err(|error| search::EvaluationError::new(error.to_string()))
    }

    fn cache_namespace(&self) -> u64 {
        BatcherClient::cache_namespace(self)
    }
}

fn run_worker(mut backend: Box<dyn InferenceBackend>, shared: Arc<Shared>) {
    let mut work_items = Vec::new();
    let mut combined = CombinedEncodedBatch::new();
    loop {
        match next_work(&shared, &mut work_items, &mut combined) {
            Work::Stop => return,
            Work::Reload(reload) => {
                let result = backend
                    .reload_weights(&reload.weights)
                    .map_err(backend_error);
                match &result {
                    Ok(()) => {
                        shared.evaluation_generation.fetch_add(1, Ordering::Release);
                        let mut pending = shared.pending.lock().unwrap();
                        pending.stats.reload_count += 1;
                        pending.stats.current_cache_namespace =
                            shared.evaluation_generation.load(Ordering::Acquire);
                    }
                    Err(error) => terminal_failure(&shared, error.clone()),
                }
                let _ = reload.tx.send(result);
            }
            Work::Evaluate => {
                let started = Instant::now();
                let result = backend.evaluate(&combined).map_err(backend_error);
                record_backend_time(&shared, started.elapsed());
                match result {
                    Ok(evaluations) if evaluations.len() == combined.len() => {
                        finish_pass(&shared, &mut work_items, evaluations)
                    }
                    Ok(_) => fail_work(
                        &shared,
                        &mut work_items,
                        BatcherError::Backend(
                            "backend returned the wrong number of evaluations".into(),
                        ),
                    ),
                    Err(error) => fail_work(&shared, &mut work_items, error),
                }
            }
        }
    }
}

fn backend_error(error: anyhow::Error) -> BatcherError {
    BatcherError::Backend(error.to_string())
}

fn next_work(
    shared: &Shared,
    items: &mut Vec<WorkItem>,
    combined: &mut CombinedEncodedBatch,
) -> Work {
    loop {
        let mut pending = shared.pending.lock().unwrap();
        pending = shared
            .cv
            .wait_while(pending, |p| !p.stop && p.commands.is_empty())
            .unwrap();
        if pending.stop {
            return Work::Stop;
        }
        if matches!(pending.commands.front(), Some(Command::Reload(_))) {
            let Some(Command::Reload(reload)) = pending.commands.pop_front()
            else {
                unreachable!()
            };
            return Work::Reload(reload);
        }
        let reload_waiting = pending
            .commands
            .iter()
            .any(|command| matches!(command, Command::Reload(_)));
        if pending.count < shared.config.preferred_batch_size && !reload_waiting {
            let wait_started = Instant::now();
            let (guard, _) = shared
                .cv
                .wait_timeout_while(pending, shared.config.max_wait, |p| {
                    !p.stop
                        && p.count < shared.config.preferred_batch_size
                        && !p
                            .commands
                            .iter()
                            .any(|command| matches!(command, Command::Reload(_)))
                })
                .unwrap();
            pending = guard;
            pending.stats.coalescing_wait_total += wait_started.elapsed();
            if pending.stop {
                return Work::Stop;
            }
            if pending.count == 0 {
                continue;
            }
        }
        items.clear();
        combined.states.clear();
        combined.legal_actions.clear();
        combined.offsets.clear();
        combined.offsets.push(0);
        while combined.len() < shared.config.max_batch_size {
            let Some(command) = pending.commands.pop_front()
            else {
                break;
            };
            let Command::Evaluate(task) = command
            else {
                pending.commands.push_front(command);
                break;
            };
            let available = shared.config.max_batch_size - combined.len();
            let end = (task.next_row + available).min(task.batch.len());
            let rows = task.next_row..end;
            if combined.append_rows(&task.batch, rows.clone()).is_err() {
                let error = BatcherError::Backend("invalid encoded evaluation batch".into());
                let _ = task.tx.send(Err(error.clone()));
                fail_pending(&mut pending, error);
                drop(pending);
                shared.cv.notify_all();
                return Work::Stop;
            }
            pending.count -= rows.len();
            let queue_wait = task.submitted_at.elapsed();
            pending.stats.queue_wait_total += queue_wait;
            pending.stats.queue_wait_max = pending.stats.queue_wait_max.max(queue_wait);
            items.push(WorkItem { task, rows });
        }
        pending.stats.inference_batches += 1;
        pending.stats.coalesced_extra_requests += items.len().saturating_sub(1) as u64;
        if items
            .iter()
            .any(|item| item.rows.end < item.task.batch.len())
        {
            pending.stats.split_batches += 1;
        }
        if combined.len() < shared.config.preferred_batch_size {
            pending.stats.partial_batches += 1;
        }
        pending.stats.lifetime_max_inference_batch = pending
            .stats
            .lifetime_max_inference_batch
            .max(combined.len() as u64);
        drop(pending);
        shared.cv.notify_all();
        return Work::Evaluate;
    }
}

fn finish_pass(shared: &Shared, items: &mut Vec<WorkItem>, evaluations: Vec<Evaluation>) {
    let mut pending = shared.pending.lock().unwrap();
    if pending.stop || pending.terminal_error.is_some() {
        let error = pending
            .terminal_error
            .clone()
            .unwrap_or(BatcherError::Shutdown);
        for item in items.drain(..) {
            let _ = item.task.tx.send(Err(error.clone()));
        }
        return;
    }
    let mut offset = 0;
    for mut item in items.drain(..) {
        let count = item.rows.len();
        item.task
            .evaluations
            .extend_from_slice(&evaluations[offset..offset + count]);
        offset += count;
        item.task.next_row = item.rows.end;
        if item.task.next_row == item.task.batch.len() {
            let _ = item.task.tx.send(Ok(EvalResponse {
                batch: item.task.batch,
                evaluations: item.task.evaluations,
            }));
        }
        else {
            // The rows left in this task stayed in `pending.count` while the
            // selected prefix was evaluated, so requeueing must not add them
            // a second time.
            pending.commands.push_front(Command::Evaluate(item.task));
        }
    }
    drop(pending);
    shared.cv.notify_all();
}

fn terminal_failure(shared: &Shared, error: BatcherError) {
    let mut pending = shared.pending.lock().unwrap();
    fail_pending(&mut pending, error);
    drop(pending);
    shared.cv.notify_all();
}

fn fail_work(shared: &Shared, items: &mut Vec<WorkItem>, error: BatcherError) {
    for item in items.drain(..) {
        let _ = item.task.tx.send(Err(error.clone()));
    }
    terminal_failure(shared, error);
}

fn fail_pending(pending: &mut Pending, error: BatcherError) {
    if pending.terminal_error.is_none() {
        pending.terminal_error = Some(error.clone());
    }
    while let Some(command) = pending.commands.pop_front() {
        match command {
            Command::Evaluate(task) => {
                let _ = task.tx.send(Err(error.clone()));
            }
            Command::Reload(reload) => {
                let _ = reload.tx.send(Err(error.clone()));
            }
        }
    }
    pending.count = 0;
}

fn record_backend_time(shared: &Shared, elapsed: Duration) {
    let mut pending = shared.pending.lock().unwrap();
    pending.stats.backend_execution_total += elapsed;
    pending.stats.backend_execution_max = pending.stats.backend_execution_max.max(elapsed);
}

#[cfg(test)]
mod tests;
