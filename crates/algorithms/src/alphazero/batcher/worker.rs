use super::super::network::{wdl_scalar, NetworkOutput};
use super::*;
use tch::{Kind, Tensor};

pub(super) struct Worker {
    vs: nn::VarStore,
    net: Network,
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
    pub(super) fn new(
        vs: nn::VarStore,
        net: Network,
        cfg: NetworkConfig,
        device: Device,
        shared: Arc<Shared>,
        precision: InferencePrecision,
    ) -> Worker {
        Worker {
            vs,
            net,
            device,
            state_size: cfg.state_size(),
            state_shape: cfg.state_shape(),
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

    pub(super) fn run(&mut self) {
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
                }
                else {
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

                    let states = pending.count as u64;
                    pending.stats.inference_batches += 1;
                    pending.stats.coalesced_extra_requests +=
                        pending.tasks.len().saturating_sub(1) as u64;
                    pending.stats.max_inference_batch =
                        pending.stats.max_inference_batch.max(states);
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
    pub(super) fn staging(
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
            }
            else {
                t
            });
        }
        slot.as_ref().unwrap().shallow_clone()
    }

    pub(super) fn process(&mut self, tasks: Vec<Task>) {
        let total: usize = tasks.iter().map(|t| t.batch.len()).sum();
        if total == 0 {
            return;
        }
        let results = tch::no_grad(|| self.evaluate(&tasks, total));

        let mut iter = results.into_iter();
        for task in tasks {
            let chunk: Vec<Evaluation> = iter.by_ref().take(task.batch.len()).collect();
            // A dropped receiver just means the client gave up; not fatal.
            let _ = task.tx.send(EvalResponse {
                batch: task.batch,
                evaluations: chunk,
            });
        }
    }

    pub(super) fn evaluate(&mut self, tasks: &[Task], total: usize) -> Vec<Evaluation> {
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
        // Keep the pinned staging buffer in FP32, then cast after the host-to-
        // device transfer when FP16 inference is enabled. `to_device_` does
        // not perform a dtype conversion, so passing `input_kind` directly
        // here leaves FP32 inputs paired with FP16 weights on CUDA.
        let batched = states_host
            .narrow(0, 0, n)
            .view([n, c, h, w])
            .to_device_(
                self.device,
                Kind::Float,
                /*non_blocking=*/ true,
                /*copy=*/ false,
            )
            .to_kind(self.input_kind);

        let (policy, value) = match self.net.forward_t(&batched, false) {
            NetworkOutput::Legacy { policy, value } => (policy, value),
            NetworkOutput::ChessAzV2 { policy, wdl } => (policy.flatten(1, -1), wdl_scalar(&wdl)),
        };
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
                // Gather before widening FP16 output. Almost every policy
                // logit is discarded, so casting the full chess policy is
                // avoidable work and memory traffic.
                let gathered_gpu = policy.gather(1, &index_gpu, false).to_kind(Kind::Float);
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
        }
        else {
            let policy = policy.to_kind(Kind::Float);
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
