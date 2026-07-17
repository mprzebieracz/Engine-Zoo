use super::*;

pub(super) fn run<G: engine_app::players::InteractiveGame>(
    args: Args,
    default_blocks: i64,
    default_filters: i64,
    self_play_fn: fn(&Batcher, &ReplayBuffer, &SelfPlayConfig) -> SelfPlayStats,
) -> Result<()> {
    let device = match args.device {
        DeviceKind::Auto => Device::cuda_if_available(),
        DeviceKind::Cuda => {
            anyhow::ensure!(
                tch::Cuda::is_available(),
                "CUDA requested but no GPU is available (check libtorch CUDA build and drivers)"
            );
            Device::Cuda(0)
        }
        DeviceKind::Cpu => Device::Cpu,
    };
    println!("device: {device:?}");
    if device.is_cuda() {
        tch::Cuda::cudnn_set_benchmark(true);
    }

    let threads = args.threads.unwrap_or_else(|| {
        if device.is_cuda() {
            128
        }
        else {
            default_cpu_threads()
        }
    });
    let wait_for = args.wait_for.unwrap_or_else(|| {
        if device.is_cuda() {
            threads.min(24)
        }
        else {
            1
        }
    });
    let self_play_precision = match args.inference_precision {
        InferencePrecisionKind::Auto => {
            if device.is_cuda() {
                InferencePrecision::Fp16
            }
            else {
                InferencePrecision::Fp32
            }
        }
        InferencePrecisionKind::Fp32 => InferencePrecision::Fp32,
        InferencePrecisionKind::Fp16 => InferencePrecision::Fp16,
    };

    let root = args
        .run_dir
        .clone()
        .unwrap_or_else(|| PathBuf::from("runs").join(G::NAME));
    let (run, cfg) = RunDir::open_or_create(&root, || RunConfig {
        game: G::NAME.into(),
        net: NetConfig::for_game::<G>(
            args.blocks.unwrap_or(default_blocks),
            args.filters.unwrap_or(default_filters),
        ),
        architecture: RunArchitecture::Legacy,
    })?;
    anyhow::ensure!(
        matches!(cfg.architecture, RunArchitecture::Legacy),
        "run {} uses chess-az-v2; use --game chess without --architecture legacy",
        root.display()
    );
    anyhow::ensure!(
        cfg.game == G::NAME,
        "run dir {} holds a {} run, not {}",
        root.display(),
        cfg.game,
        G::NAME
    );

    let mut vs = nn::VarStore::new(device);
    let net = AlphaZeroNet::new(&vs.root(), &cfg.net);
    let mut next_ckpt = run.latest_checkpoint().map_or(1, |(idx, _)| idx + 1);
    match run.training_checkpoint() {
        Some(path) => {
            println!("resuming from {}", path.display());
            vs.load(&path)?;
        }
        None => {
            vs.save(run.checkpoint_path(0))?;
        }
    }
    if !run.best_path().exists() {
        vs.save(run.best_path())?;
    }

    let replay = ReplayBuffer::new(args.buffer, G::state_size(), G::ACTION_SIZE);
    let train_cfg = TrainConfig {
        micro_batch_size: args.batch_size,
        batch_size: args.minibatch_size,
        train_steps: args.train_steps,
        progress_every: args.train_progress_every,
        lr: args.lr,
        weight_decay: args.weight_decay,
    };
    let mut opt = algorithms::alphazero::build_optimizer(&vs, &train_cfg)?;
    let sp_cfg = SelfPlayConfig {
        num_games: args.games,
        threads,
        max_moves: args.max_moves,
        progress_every: args.progress_every,
        tt_entries: args.tt_entries,
        fast_simulations: args
            .fast_simulations
            .unwrap_or_else(|| (args.simulations / 8).max(1)),
        full_simulation_probability: args.full_simulation_probability,
        resignation_enabled: !args.disable_resignation,
        resignation_threshold: args.resignation_threshold,
        resignation_consecutive_moves: args.resignation_consecutive_moves,
        resignation_min_ply: args.resignation_min_ply,
        resignation_disable_probability: args.resignation_disable_probability,
        mcts: MctsConfig {
            simulations: args.simulations,
            leaf_batch_size: args.mcts_leaf_batch_size.max(1),
            fpu_reduction: args.fpu_reduction,
            variant: match args.mcts_variant {
                SearchKind::Puct => MctsVariant::Puct,
                SearchKind::Gumbel => MctsVariant::Gumbel {
                    sampled_actions: args.gumbel_sampled_actions,
                },
            },
            ..Default::default()
        },
        ..Default::default()
    };

    let archive_interval = (args.archive_checkpoint_minutes > 0)
        .then(|| Duration::from_secs(args.archive_checkpoint_minutes * 60));
    let mut last_archive = Instant::now();
    let self_play_batcher = Batcher::new_with_precision(
        &cfg.net,
        &run.best_path(),
        device,
        wait_for,
        Duration::from_millis(args.batch_timeout_ms),
        self_play_precision,
    )?;
    let mut iteration = 0usize;
    while args.forever || iteration < args.iterations {
        println!("=== iteration {iteration} ===");
        let batcher_before = self_play_batcher.stats();
        let self_play_started = Instant::now();
        let self_play_stats = self_play_fn(&self_play_batcher, &replay, &sp_cfg);
        let self_play_secs = self_play_started.elapsed().as_secs_f64();
        let batcher_stats = self_play_batcher.stats().since(batcher_before);
        let games_per_sec = self_play_stats.games as f64 / self_play_secs.max(f64::EPSILON);
        let positions_per_sec = self_play_stats.moves as f64 / self_play_secs.max(f64::EPSILON);
        let tt_queries = self_play_stats.tt_hits + self_play_stats.tt_misses;
        let tt_hit_rate = if tt_queries == 0 {
            0.0
        }
        else {
            self_play_stats.tt_hits as f64 / tt_queries as f64
        };
        println!(
            "self-play: {} games in {self_play_secs:.1}s ({:.1} games/s, {:.1} moves/game)",
            args.games,
            args.games as f64 / self_play_secs,
            self_play_stats.avg_moves_per_game()
        );
        println!(
            "self-play stats: full={} fast={} resignations={} tt={:.1}%",
            self_play_stats.full_searches,
            self_play_stats.fast_searches,
            self_play_stats.resignations,
            100.0 * tt_hit_rate
        );
        println!(
            "batcher stats: requests={} states={} inference_batches={} avg_batch={:.1} max_batch_highwater={}",
            batcher_stats.submitted_batches,
            batcher_stats.submitted_states,
            batcher_stats.inference_batches,
            batcher_stats.submitted_states as f64
                / batcher_stats.inference_batches.max(1) as f64,
            batcher_stats.max_inference_batch,
        );

        let train_started = Instant::now();
        let metrics = train(&net, &mut opt, &replay, device, &cfg.net, &train_cfg);
        let train_secs = train_started.elapsed().as_secs_f64();
        println!("train: {train_secs:.1}s");

        let mut record = json!({
            "iteration": iteration,
            "mode": match args.mode { Mode::Continuous => "continuous", Mode::Gated => "gated" },
            "mcts_variant": match args.mcts_variant { SearchKind::Puct => "puct", SearchKind::Gumbel => "gumbel" },
            "buffer_size": replay.len(),
            "self_play_secs": self_play_secs,
            "games_per_sec": games_per_sec,
            "positions_per_sec": positions_per_sec,
            "train_secs": train_secs,
            "games": args.games,
            "self_play_games": self_play_stats.games,
            "self_play_moves": self_play_stats.moves,
            "avg_moves_per_game": self_play_stats.avg_moves_per_game(),
            "full_searches": self_play_stats.full_searches,
            "fast_searches": self_play_stats.fast_searches,
            "resignations": self_play_stats.resignations,
            "tt_hits": self_play_stats.tt_hits,
            "tt_misses": self_play_stats.tt_misses,
            "tt_inserts": self_play_stats.tt_inserts,
            "tt_hit_rate": tt_hit_rate,
            "fpu_reduction": args.fpu_reduction,
            "mcts_leaf_batch_size": args.mcts_leaf_batch_size.max(1),
            "threads": threads,
            "wait_for": wait_for,
            "batch_timeout_ms": args.batch_timeout_ms,
            "tt_entries": args.tt_entries,
            "fast_simulations": sp_cfg.fast_simulations,
            "full_simulation_probability": sp_cfg.full_simulation_probability,
            "resignation_enabled": sp_cfg.resignation_enabled,
            "resignation_threshold": sp_cfg.resignation_threshold,
            "resignation_min_ply": sp_cfg.resignation_min_ply,
            "resignation_consecutive_moves": sp_cfg.resignation_consecutive_moves,
            "resignation_disable_probability": sp_cfg.resignation_disable_probability,
            "self_play_precision": match self_play_precision { InferencePrecision::Fp32 => "fp32", InferencePrecision::Fp16 => "fp16" },
            "submitted_batches": batcher_stats.submitted_batches,
            "submitted_states": batcher_stats.submitted_states,
            "inference_batches": batcher_stats.inference_batches,
            "coalesced_extra_requests": batcher_stats.coalesced_extra_requests,
            "avg_inference_batch": batcher_stats.submitted_states as f64 / batcher_stats.inference_batches.max(1) as f64,
            "max_submitted_batch_highwater": batcher_stats.max_submitted_batch,
            "max_inference_batch_highwater": batcher_stats.max_inference_batch,
        });
        record["train_steps"] = metrics.as_ref().map_or(0, |m| m.train_steps).into();
        if let Some(m) = &metrics {
            record["policy_loss"] = m.policy_loss.into();
            record["value_loss"] = m.value_loss.into();
        }

        match args.mode {
            Mode::Continuous => {
                let checkpoint_started = Instant::now();
                let numbered_checkpoint =
                    save_numbered_checkpoint(&run, &vs, next_ckpt, args.numbered_checkpoint_every)?;
                vs.save(run.best_path())?;
                self_play_batcher.reload_weights(&run.best_path())?;
                next_ckpt += 1;
                record["checkpoint"] = numbered_checkpoint
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .into();
                record["checkpoint_secs"] = checkpoint_started.elapsed().as_secs_f64().into();
            }
            Mode::Gated => {
                vs.save(run.candidate_path())?;
                let arena_cfg = ArenaConfig {
                    games: args.gate_games,
                    max_moves: args.max_moves,
                    ..Default::default()
                };
                let wait_for = 1;
                let mut candidate = engine_app::players::AlphaZeroAgent::<G>::new(
                    &cfg.net,
                    &run.candidate_path(),
                    device,
                    args.gate_simulations,
                    wait_for,
                    BATCH_TIMEOUT,
                )?;
                let mut baseline = engine_app::players::AlphaZeroAgent::<G>::new(
                    &cfg.net,
                    &run.best_path(),
                    device,
                    args.gate_simulations,
                    wait_for,
                    BATCH_TIMEOUT,
                )?;
                let arena_result = arena::evaluate::<G>(&mut candidate, &mut baseline, &arena_cfg)?;
                let winrate = arena_result.score();

                let promoted = winrate >= args.gate_threshold;
                record["arena_winrate"] = winrate.into();
                record["gate_threshold"] = args.gate_threshold.into();
                record["promoted"] = promoted.into();
                if promoted {
                    println!(
                        "candidate promoted: {:.1}% >= {:.1}%",
                        100.0 * winrate,
                        100.0 * args.gate_threshold
                    );
                    let numbered_checkpoint = save_numbered_checkpoint(
                        &run,
                        &vs,
                        next_ckpt,
                        args.numbered_checkpoint_every,
                    )?;
                    vs.save(run.best_path())?;
                    self_play_batcher.reload_weights(&run.best_path())?;
                    next_ckpt += 1;
                    record["checkpoint"] = numbered_checkpoint
                        .as_ref()
                        .map(|path| path.display().to_string())
                        .into();
                }
                else {
                    println!(
                        "candidate rejected: {:.1}% < {:.1}%; self-play keeps the old best",
                        100.0 * winrate,
                        100.0 * args.gate_threshold
                    );
                    restore_best_after_rejection(&mut vs, &mut opt, &run, &train_cfg)?;
                }
            }
        }

        if let Some(interval) = archive_interval {
            if last_archive.elapsed() >= interval {
                let archive_started = Instant::now();
                let unix_secs = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs();
                let archive_path = run.archive_checkpoint_path(iteration, unix_secs)?;
                vs.save(&archive_path)?;
                println!("archived checkpoint: {}", archive_path.display());
                record["archive_checkpoint"] = archive_path.display().to_string().into();
                record["archive_checkpoint_secs"] = archive_started.elapsed().as_secs_f64().into();
                last_archive = Instant::now();
            }
        }

        run.log_metrics(record)?;
        iteration += 1;
    }
    Ok(())
}

fn restore_best_after_rejection(
    vs: &mut nn::VarStore,
    optimizer: &mut nn::Optimizer,
    run: &RunDir,
    cfg: &TrainConfig,
) -> Result<()> {
    vs.load(run.best_path())?;
    // Adam moments belong to the rejected weights too. Keeping them after a
    // rollback would bias the next candidate away from the accepted baseline.
    *optimizer = algorithms::alphazero::build_optimizer(vs, cfg)?;
    Ok(())
}

pub(super) fn save_numbered_checkpoint(
    run: &RunDir,
    vs: &nn::VarStore,
    next_ckpt: u32,
    every: u32,
) -> Result<Option<PathBuf>> {
    if every == 0 || !next_ckpt.is_multiple_of(every) {
        return Ok(None);
    }
    let path = run.checkpoint_path(next_ckpt);
    vs.save(&path)?;
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_root() -> PathBuf {
        std::env::temp_dir().join(format!(
            "engine-zoo-gate-rollback-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn rejected_candidate_restores_the_accepted_weights() {
        let root = test_root();
        let cfg = NetConfig::for_game::<Connect4>(0, 2);
        let (run, _) = RunDir::open_or_create(&root, || RunConfig {
            game: Connect4::NAME.into(),
            net: cfg.clone(),
            architecture: RunArchitecture::Legacy,
        })
        .unwrap();
        let train_cfg = TrainConfig {
            micro_batch_size: 1,
            batch_size: 1,
            train_steps: 1,
            progress_every: 0,
            ..TrainConfig::default()
        };

        let mut vs = nn::VarStore::new(Device::Cpu);
        let _net = AlphaZeroNet::new(&vs.root(), &cfg);
        vs.save(run.best_path()).unwrap();
        let mut optimizer = algorithms::alphazero::build_optimizer(&vs, &train_cfg).unwrap();

        tch::no_grad(|| {
            for mut tensor in vs.variables().into_values() {
                let _ = tensor.fill_(1.0);
            }
        });
        restore_best_after_rejection(&mut vs, &mut optimizer, &run, &train_cfg).unwrap();

        let mut expected_vs = nn::VarStore::new(Device::Cpu);
        let _expected_net = AlphaZeroNet::new(&expected_vs.root(), &cfg);
        expected_vs.load(run.best_path()).unwrap();
        let expected = expected_vs.variables();
        for (name, restored) in vs.variables() {
            assert!(
                restored.allclose(expected.get(&name).unwrap(), 0.0, 0.0, false),
                "{name} was not restored from best"
            );
        }

        fs::remove_dir_all(root).unwrap();
    }
}
