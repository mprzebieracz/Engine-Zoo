use super::evaluator::{EncodedEvalBatch, EncodedEvaluator, Evaluation};
use super::network::{NetConfig, Network, NetworkConfig};
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
#[cfg(test)]
use tch::Kind;
use tch::{nn, Device};

mod worker;
use worker::Worker;

/// Cross-thread dynamic batching for network inference: many search threads
/// submit small eval batches; a single worker thread coalesces everything
/// that arrives within `timeout` (or as soon as `wait_for_count` states are
/// pending) into one forward pass.
///
/// Two things keep the GPU path fast:
/// - staging buffers are *pinned* host memory (grow-only, reused across
///   batches), so host<->device transfers are DMA instead of pageable copies;
/// - legal-action logits are gathered *on device* before the D2H copy — for
///   chess that shrinks the transferred policy from 20480 floats per state to
///   the ~40 that are actually read (~500x less PCIe traffic).
pub struct Batcher {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<()>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferencePrecision {
    Fp32,
    Fp16,
}

/// Cheap-to-clone handle used by search threads to submit work.
#[derive(Clone)]
pub struct BatcherClient {
    shared: Arc<Shared>,
}

/// Cumulative dynamic-batching counters. Taking two snapshots around a
/// self-play round gives the exact request and coalescing totals for it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BatcherStats {
    pub submitted_batches: u64,
    pub submitted_states: u64,
    pub inference_batches: u64,
    /// Requests beyond the first that shared an inference pass.
    pub coalesced_extra_requests: u64,
    pub max_submitted_batch: u64,
    pub max_inference_batch: u64,
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
            // Maxima are cumulative high-water marks and cannot be differenced.
            max_submitted_batch: self.max_submitted_batch,
            max_inference_batch: self.max_inference_batch,
        }
    }
}

struct Task {
    batch: EncodedEvalBatch,
    tx: SyncSender<EvalResponse>,
}

struct EvalResponse {
    batch: EncodedEvalBatch,
    evaluations: Vec<Evaluation>,
}

struct Reload {
    weights: PathBuf,
    tx: SyncSender<Result<()>>,
}

struct Pending {
    tasks: Vec<Task>,
    reloads: VecDeque<Reload>,
    /// Total states across `tasks`.
    count: usize,
    stop: bool,
    stats: BatcherStats,
}

impl Pending {
    fn new() -> Self {
        Pending {
            tasks: Vec::new(),
            reloads: VecDeque::new(),
            count: 0,
            stop: false,
            stats: BatcherStats::default(),
        }
    }
}

struct Shared {
    pending: Mutex<Pending>,
    cv: Condvar,
    wait_for_count: usize,
    timeout: Duration,
}

enum Work {
    Tasks(Vec<Task>),
    Reload(Reload),
}

impl Batcher {
    /// Builds the network from `cfg`, loads `weights`, and starts the worker.
    pub fn new(
        cfg: &NetConfig,
        weights: &Path,
        device: Device,
        wait_for_count: usize,
        timeout: Duration,
    ) -> Result<Batcher> {
        Self::new_with_precision(
            cfg,
            weights,
            device,
            wait_for_count,
            timeout,
            InferencePrecision::Fp32,
        )
    }

    pub fn new_with_precision(
        cfg: &NetConfig,
        weights: &Path,
        device: Device,
        wait_for_count: usize,
        timeout: Duration,
        precision: InferencePrecision,
    ) -> Result<Batcher> {
        Self::new_with_network_precision(
            &NetworkConfig::Legacy(cfg.clone()),
            weights,
            device,
            wait_for_count,
            timeout,
            precision,
        )
    }

    /// Builds the explicitly selected network architecture. Callers loading a
    /// run config should use this instead of inferring an architecture from
    /// checkpoint tensor shapes.
    pub fn new_with_network(
        cfg: &NetworkConfig,
        weights: &Path,
        device: Device,
        wait_for_count: usize,
        timeout: Duration,
    ) -> Result<Batcher> {
        Self::new_with_network_precision(
            cfg,
            weights,
            device,
            wait_for_count,
            timeout,
            InferencePrecision::Fp32,
        )
    }

    pub fn new_with_network_precision(
        cfg: &NetworkConfig,
        weights: &Path,
        device: Device,
        wait_for_count: usize,
        timeout: Duration,
        precision: InferencePrecision,
    ) -> Result<Batcher> {
        anyhow::ensure!(
            precision == InferencePrecision::Fp32 || device.is_cuda(),
            "FP16 inference is only supported on CUDA"
        );
        let mut vs = nn::VarStore::new(device);
        let net = Network::new(&vs.root(), cfg);
        vs.load(weights)
            .with_context(|| format!("loading network weights from {}", weights.display()))?;
        if precision == InferencePrecision::Fp16 {
            vs.half();
        }

        let shared = Arc::new(Shared {
            pending: Mutex::new(Pending::new()),
            cv: Condvar::new(),
            wait_for_count: wait_for_count.max(1),
            timeout,
        });

        let worker_shared = Arc::clone(&shared);
        let cfg = cfg.clone();
        let worker = std::thread::Builder::new()
            .name("batcher".into())
            .spawn(move || {
                Worker::new(vs, net, cfg, device, worker_shared, precision).run();
            })
            .context("spawning batcher worker")?;

        Ok(Batcher {
            shared,
            worker: Some(worker),
        })
    }

    pub fn client(&self) -> BatcherClient {
        BatcherClient {
            shared: Arc::clone(&self.shared),
        }
    }

    pub fn stats(&self) -> BatcherStats {
        self.shared.pending.lock().unwrap().stats
    }

    /// Reloads network weights inside the existing worker, preserving the
    /// worker thread and grow-only staging buffers between self-play rounds.
    pub fn reload_weights(&self, weights: &Path) -> Result<()> {
        let (tx, rx) = sync_channel(1);
        {
            let mut pending = self.shared.pending.lock().unwrap();
            anyhow::ensure!(!pending.stop, "batcher already shut down");
            pending.reloads.push_back(Reload {
                weights: weights.to_path_buf(),
                tx,
            });
        }
        self.shared.cv.notify_all();
        rx.recv()
            .context("batcher worker stopped before reloading weights")?
    }
}

#[cfg(test)]
mod tests;

impl Drop for Batcher {
    fn drop(&mut self) {
        self.shared.pending.lock().unwrap().stop = true;
        self.shared.cv.notify_all();
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl EncodedEvaluator for BatcherClient {
    fn evaluate(&mut self, batch: &mut EncodedEvalBatch) -> Vec<Evaluation> {
        let n = batch.len();
        if n == 0 {
            return Vec::new();
        }
        // Transfer the caller's reusable allocations across the worker
        // boundary. The response returns the same vectors for the next search
        // request, avoiding three allocations and full-vector copies here.
        let owned = std::mem::take(batch);

        let (tx, rx) = sync_channel(1);
        {
            let mut pending = self.shared.pending.lock().unwrap();
            assert!(!pending.stop, "batcher already shut down");
            pending.stats.submitted_batches += 1;
            pending.stats.submitted_states += n as u64;
            pending.stats.max_submitted_batch = pending.stats.max_submitted_batch.max(n as u64);
            pending.tasks.push(Task { batch: owned, tx });
            pending.count += n;
        }
        self.shared.cv.notify_all();
        let response = rx
            .recv()
            .expect("batcher worker died (it panics only on inference errors)");
        *batch = response.batch;
        response.evaluations
    }
}
