use super::evaluator::{EvalBatch, Evaluation, Evaluator};
use super::network::{AlphaZeroNet, NetConfig};
use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;
use tch::{nn, Device, Kind, Tensor};

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

struct Task {
    batch: EvalBatch,
    tx: SyncSender<Vec<Evaluation>>,
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
}

impl Pending {
    fn new() -> Self {
        Pending {
            tasks: Vec::new(),
            reloads: VecDeque::new(),
            count: 0,
            stop: false,
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
        anyhow::ensure!(
            precision == InferencePrecision::Fp32 || device.is_cuda(),
            "FP16 inference is only supported on CUDA"
        );
        let mut vs = nn::VarStore::new(device);
        let net = AlphaZeroNet::new(&vs.root(), cfg);
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

impl Evaluator for BatcherClient {
    fn evaluate(&mut self, batch: &EvalBatch) -> Vec<Evaluation> {
        let n = batch.len();
        if n == 0 {
            return Vec::new();
        }
        // The submitted data must outlive this call on another thread, so it
        // is copied out of the caller's reusable buffers.
        let owned = EvalBatch {
            states: batch.states.clone(),
            legal: batch.legal.clone(),
            offsets: batch.offsets.clone(),
        };

        let (tx, rx) = sync_channel(1);
        {
            let mut pending = self.shared.pending.lock().unwrap();
            assert!(!pending.stop, "batcher already shut down");
            pending.tasks.push(Task { batch: owned, tx });
            pending.count += n;
        }
        self.shared.cv.notify_all();
        rx.recv()
            .expect("batcher worker died (it panics only on inference errors)")
    }
}

struct Worker {
    vs: nn::VarStore,
    net: AlphaZeroNet,
    device: Device,
    state_size: usize,
    state_shape: [i64; 3],
    input_kind: Kind,
    shared: Arc<Shared>,
    // Grow-only staging buffers (pinned when on CUDA). Rows beyond the
    // current batch are stale garbage — only the first `n` rows are ever
    // read, and index padding points at action 0.
    states_buf: Option<Tensor>,
    index_buf: Option<Tensor>,
    gathered_buf: Option<Tensor>,
    value_buf: Option<Tensor>,
    precision: InferencePrecision,
}

impl Worker {
    fn new(
        vs: nn::VarStore,
        net: AlphaZeroNet,
        cfg: NetConfig,
        device: Device,
        shared: Arc<Shared>,
        precision: InferencePrecision,
    ) -> Worker {
        Worker {
            vs,
            net,
            device,
            state_size: (cfg.input_channels * cfg.height * cfg.width) as usize,
            state_shape: [cfg.input_channels, cfg.height, cfg.width],
            input_kind: match precision {
                InferencePrecision::Fp32 => Kind::Float,
                InferencePrecision::Fp16 => Kind::Half,
            },
            shared,
            states_buf: None,
            index_buf: None,
            gathered_buf: None,
            value_buf: None,
            precision,
        }
    }

    fn run(&mut self) {
        loop {
            let work = {
                let mut pending = self.shared.pending.lock().unwrap();
                pending = self
                    .shared
                    .cv
                    .wait_while(pending, |p| !p.stop && p.count == 0 && p.reloads.is_empty())
                    .unwrap();

                if pending.stop && pending.tasks.is_empty() && pending.reloads.is_empty() {
                    return;
                }

                if pending.tasks.is_empty() {
                    Work::Reload(
                        pending
                            .reloads
                            .pop_front()
                            .expect("worker woke without eval tasks only when reload is queued"),
                    )
                } else {
                    if pending.count < self.shared.wait_for_count && !pending.stop {
                        let (guard, _) = self
                            .shared
                            .cv
                            .wait_timeout_while(pending, self.shared.timeout, |p| {
                                !p.stop
                                    && p.count < self.shared.wait_for_count
                                    && p.reloads.is_empty()
                            })
                            .unwrap();
                        pending = guard;
                    }

                    pending.count = 0;
                    Work::Tasks(std::mem::take(&mut pending.tasks))
                }
            };

            match work {
                Work::Tasks(tasks) if !tasks.is_empty() => self.process(tasks),
                Work::Tasks(_) => {}
                Work::Reload(reload) => {
                    let result = self.reload_weights(&reload.weights);
                    let _ = reload.tx.send(result);
                }
            }
        }
    }

    fn reload_weights(&mut self, weights: &Path) -> Result<()> {
        self.vs.float();
        self.vs
            .load(weights)
            .with_context(|| format!("loading network weights from {}", weights.display()))?;
        if self.precision == InferencePrecision::Fp16 {
            self.vs.half();
        }
        Ok(())
    }

    /// Allocates (or grows) a staging buffer, pinned when running on CUDA.
    fn staging(
        slot: &mut Option<Tensor>,
        rows: i64,
        cols: i64,
        kind: Kind,
        device: Device,
    ) -> Tensor {
        let needs_alloc = match slot {
            Some(t) => t.size()[0] < rows || t.size()[1] < cols,
            None => true,
        };
        if needs_alloc {
            let old_cols = slot.as_ref().map_or(0, |t| t.size()[1]);
            let t = Tensor::zeros([rows, cols.max(old_cols)], (kind, Device::Cpu));
            *slot = Some(if device.is_cuda() {
                t.pin_memory(device)
            } else {
                t
            });
        }
        slot.as_ref().unwrap().shallow_clone()
    }

    fn process(&mut self, tasks: Vec<Task>) {
        let total: usize = tasks.iter().map(|t| t.batch.len()).sum();
        if total == 0 {
            return;
        }
        let results = tch::no_grad(|| self.evaluate(&tasks, total));

        let mut iter = results.into_iter();
        for task in tasks {
            let chunk: Vec<Evaluation> = iter.by_ref().take(task.batch.len()).collect();
            // A dropped receiver just means the client gave up; not fatal.
            let _ = task.tx.send(chunk);
        }
    }

    fn evaluate(&mut self, tasks: &[Task], total: usize) -> Vec<Evaluation> {
        let n = total as i64;
        let cuda = self.device.is_cuda();
        let min_rows = (self.shared.wait_for_count * 2).max(total) as i64;

        // Stage all canonical states contiguously and ship them to the device.
        let states_host = Self::staging(
            &mut self.states_buf,
            min_rows,
            self.state_size as i64,
            Kind::Float,
            self.device,
        );
        {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    states_host.data_ptr() as *mut f32,
                    (states_host.size()[0] * states_host.size()[1]) as usize,
                )
            };
            let mut offset = 0;
            for task in tasks {
                dst[offset..offset + task.batch.states.len()].copy_from_slice(&task.batch.states);
                offset += task.batch.states.len();
            }
        }
        let [c, h, w] = self.state_shape;
        let batched = states_host.narrow(0, 0, n).view([n, c, h, w]).to_device_(
            self.device,
            self.input_kind,
            /*non_blocking=*/ true,
            /*copy=*/ false,
        );

        let (policy, value) = self.net.forward_t(&batched, false);
        let policy = policy.to_kind(Kind::Float);
        let value = value.to_kind(Kind::Float);

        let max_actions = tasks
            .iter()
            .flat_map(|t| t.batch.offsets.windows(2))
            .map(|w| (w[1] - w[0]) as i64)
            .max()
            .unwrap_or(0);

        // Per-state legal-action logits. On CUDA, gather them on device and
        // copy only those; on CPU just read from the full policy tensor.
        let mut out = Vec::with_capacity(total);
        if cuda {
            let mut gathered_rows: Option<(*const f32, usize)> = None; // (data, row stride)
            if max_actions > 0 {
                let index_host = Self::staging(
                    &mut self.index_buf,
                    min_rows,
                    max_actions,
                    Kind::Int64,
                    self.device,
                );
                let gathered_host = Self::staging(
                    &mut self.gathered_buf,
                    min_rows,
                    max_actions,
                    Kind::Float,
                    self.device,
                );
                // Use the buffers' actual width (they only ever grow): slicing
                // a wider buffer down to max_actions would make the D2H copy's
                // destination non-contiguous, silently degrading it from a
                // pinned DMA transfer to a pageable copy (~1000x slower).
                let width = gathered_host.size()[1];

                {
                    let idx = unsafe {
                        std::slice::from_raw_parts_mut(
                            index_host.data_ptr() as *mut i64,
                            (index_host.size()[0] * index_host.size()[1]) as usize,
                        )
                    };
                    let mut row = 0;
                    for task in tasks {
                        for s in 0..task.batch.len() {
                            let begin = task.batch.offsets[s] as usize;
                            let end = task.batch.offsets[s + 1] as usize;
                            let dst = &mut idx[row * width as usize..(row + 1) * width as usize];
                            dst.fill(0); // padding gathers action 0, never read
                            for (k, &a) in task.batch.legal[begin..end].iter().enumerate() {
                                dst[k] = i64::from(a);
                            }
                            row += 1;
                        }
                    }
                }

                let index_gpu = index_host.narrow(0, 0, n).narrow(1, 0, width).to_device_(
                    self.device,
                    Kind::Int64,
                    true,
                    false,
                );
                let gathered_gpu = policy.gather(1, &index_gpu, false);
                // Synchronous D2H into pinned memory: fast DMA, data valid on return.
                gathered_host
                    .narrow(0, 0, n)
                    .narrow(1, 0, width)
                    .copy_(&gathered_gpu);

                gathered_rows = Some((gathered_host.data_ptr() as *const f32, width as usize));
            }

            let value_host =
                Self::staging(&mut self.value_buf, min_rows, 1, Kind::Float, self.device);
            value_host.narrow(0, 0, n).copy_(&value.view([n, 1]));
            let values =
                unsafe { std::slice::from_raw_parts(value_host.data_ptr() as *const f32, total) };

            let mut row = 0;
            for task in tasks {
                for s in 0..task.batch.len() {
                    let count = (task.batch.offsets[s + 1] - task.batch.offsets[s]) as usize;
                    let logits = match gathered_rows {
                        Some((data, width)) => unsafe {
                            std::slice::from_raw_parts(data.add(row * width), count).to_vec()
                        },
                        None => Vec::new(),
                    };
                    out.push(Evaluation {
                        logits,
                        value: values[row],
                    });
                    row += 1;
                }
            }
        } else {
            let policy_flat: Vec<f32> = policy.contiguous().view(-1).try_into().unwrap();
            let values: Vec<f32> = value.contiguous().view(-1).try_into().unwrap();
            let action_size = policy.size()[1] as usize;

            let mut row = 0;
            for task in tasks {
                for s in 0..task.batch.len() {
                    let begin = task.batch.offsets[s] as usize;
                    let end = task.batch.offsets[s + 1] as usize;
                    let logits = task.batch.legal[begin..end]
                        .iter()
                        .map(|&a| policy_flat[row * action_size + a as usize])
                        .collect();
                    out.push(Evaluation {
                        logits,
                        value: values[row],
                    });
                    row += 1;
                }
            }
        }
        out
    }
}
