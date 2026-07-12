use super::*;

pub(super) fn run_chess_az_v2(
    args: Args,
    run: RunDir,
    cfg: RunConfig,
    v2: ChessAzV2Config,
) -> Result<()> {
    anyhow::ensure!(
        matches!(args.mode, Mode::Continuous),
        "gated training is not yet supported for chess-az-v2"
    );
    let device = match args.device {
        DeviceKind::Auto => Device::cuda_if_available(),
        DeviceKind::Cuda => {
            anyhow::ensure!(
                tch::Cuda::is_available(),
                "CUDA requested but no GPU is available"
            );
            Device::Cuda(0)
        }
        DeviceKind::Cpu => Device::Cpu,
    };
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
    // The v2 trainer keeps `vs` and its optimizer in FP32. This precision
    // selection applies only to the self-play inference batcher.
    let self_play_precision = match args.inference_precision {
        InferencePrecisionKind::Auto if device.is_cuda() => InferencePrecision::Fp16,
        InferencePrecisionKind::Auto | InferencePrecisionKind::Fp32 => InferencePrecision::Fp32,
        InferencePrecisionKind::Fp16 => InferencePrecision::Fp16,
    };
    let network_cfg = cfg.network_config();
    anyhow::ensure!(
        matches!(&network_cfg, NetworkConfig::ChessAzV2(saved) if *saved == v2),
        "saved run configuration is not the requested chess-az-v2 architecture"
    );
    let mut vs = nn::VarStore::new(device);
    let net = ChessAzV2Net::new(&vs.root(), v2);
    let mut next_ckpt = run.latest_checkpoint().map_or(1, |(index, _)| index + 1);
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
    let replay = ReplayBuffer::new(
        args.buffer,
        v2.state_size(),
        ChessAzV2Config::ACTION_SIZE as usize,
    );
    let train_cfg = TrainConfig {
        micro_batch_size: args.batch_size,
        batch_size: args.minibatch_size,
        train_steps: args.train_steps,
        progress_every: args.train_progress_every,
        lr: args.lr,
        weight_decay: args.weight_decay,
    };
    let mut optimizer = algorithms::alphazero::build_optimizer(&vs, &train_cfg)?;
    let profiles = ChessV2GumbelProfiles {
        full: GumbelSearchProfile::new(args.v2_full_simulations, args.v2_full_root_candidates),
        fast: GumbelSearchProfile::new(args.v2_fast_simulations, args.v2_fast_root_candidates),
    };
    let mut self_play_cfg = SelfPlayConfig::chess_v2_defaults();
    self_play_cfg.num_games = args.games;
    self_play_cfg.threads = threads;
    self_play_cfg.max_moves = args.max_moves;
    self_play_cfg.progress_every = args.progress_every;
    self_play_cfg.tt_entries = args.tt_entries;
    self_play_cfg.full_simulation_probability = args.v2_full_simulation_probability;
    self_play_cfg.chess_v2_gumbel_profiles = Some(profiles);
    self_play_cfg.mcts.leaf_batch_size = args.mcts_leaf_batch_size.max(1);
    self_play_cfg.mcts.fpu_reduction = args.fpu_reduction;
    self_play_cfg.resignation_enabled = !args.disable_resignation;
    self_play_cfg.resignation_threshold = args.resignation_threshold;
    self_play_cfg.resignation_consecutive_moves = args.resignation_consecutive_moves;
    self_play_cfg.resignation_min_ply = args.resignation_min_ply;
    self_play_cfg.resignation_disable_probability = args.resignation_disable_probability;
    self_play_cfg
        .validate_chess_v2()
        .map_err(anyhow::Error::msg)?;
    let batcher = Batcher::new_with_network_precision(
        &network_cfg,
        &run.best_path(),
        device,
        wait_for,
        Duration::from_millis(args.batch_timeout_ms),
        self_play_precision,
    )?;

    let mut iteration = 0;
    while args.forever || iteration < args.iterations {
        let stats = match v2.history {
            1 => self_play_chess_az_v2::<1>(&batcher, &replay, &self_play_cfg),
            4 => self_play_chess_az_v2::<4>(&batcher, &replay, &self_play_cfg),
            8 => self_play_chess_az_v2::<8>(&batcher, &replay, &self_play_cfg),
            _ => unreachable!("validated ChessAzV2Config history"),
        };
        let metrics = train_chess_az_v2(&net, &mut optimizer, &replay, device, v2, &train_cfg);
        let checkpoint =
            save_numbered_checkpoint(&run, &vs, next_ckpt, args.numbered_checkpoint_every)?;
        vs.save(run.best_path())?;
        batcher.reload_weights(&run.best_path())?;
        next_ckpt += 1;
        let mut record = json!({
            "iteration": iteration,
            "architecture": "chess-az-v2",
            "history": v2.history,
            "self_play_precision": match self_play_precision {
                InferencePrecision::Fp16 => "fp16",
                InferencePrecision::Fp32 => "fp32",
            },
            "training_precision": "fp32",
            "self_play_games": stats.games,
            "self_play_moves": stats.moves,
            "buffer_size": replay.len(),
            "full_searches": stats.full_searches,
            "fast_searches": stats.fast_searches,
            "checkpoint": checkpoint.map(|p| p.display().to_string()),
        });
        if let Some(metrics) = metrics {
            record["policy_loss"] = metrics.policy_loss.into();
            record["value_loss"] = metrics.value_loss.into();
        }
        run.log_metrics(record)?;
        iteration += 1;
    }
    Ok(())
}
