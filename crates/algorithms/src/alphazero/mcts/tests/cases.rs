use super::*;

#[test]
fn terminal_root_returns_reward_without_evaluator() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls.clone()),
        MctsConfig {
            simulations: 4,
            eps: 0.0,
            ..Default::default()
        },
    );

    let result = mcts.search(&TerminalGame);

    assert_eq!(result.value, -1.0);
    assert_eq!(result.policy, vec![0.0, 0.0]);
    assert!(calls.lock().unwrap().is_empty());
}

#[test]
fn puct_finds_forced_terminal_win() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls),
        MctsConfig {
            simulations: 32,
            eps: 0.0,
            ..Default::default()
        },
    );

    let result = mcts.search(&ImmediateOutcomeGame::default());

    assert_eq!(result.best_action(), 0);
    assert!(result.policy[0] > result.policy[1], "{:?}", result.policy);
    assert!(result.policy[0] > result.policy[2], "{:?}", result.policy);
}

#[test]
fn single_leaf_search_evaluates_one_state_per_call() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls.clone()),
        MctsConfig {
            simulations: 4,
            eps: 0.0,
            ..Default::default()
        },
    );

    let result = mcts.search(&SinglePathGame::default());

    assert_eq!(result.policy, vec![1.0]);
    assert_eq!(*calls.lock().unwrap(), vec![1, 1]);
}

#[test]
fn batched_puct_evaluates_multiple_states_per_call() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls.clone()),
        MctsConfig {
            simulations: 8,
            leaf_batch_size: 4,
            eps: 0.0,
            ..Default::default()
        },
    );

    let result = mcts.search(&Connect4::default());

    assert!((result.policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert_eq!(mcts.virtual_loss_total(), 0);
    assert!(
        calls.lock().unwrap().iter().any(|&n| n > 1),
        "expected at least one batched evaluator call, got {:?}",
        calls.lock().unwrap()
    );
}

#[test]
fn duplicate_batched_leaves_are_evaluated_once_and_backed_up_each_time() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls.clone()),
        MctsConfig {
            simulations: 4,
            leaf_batch_size: 4,
            eps: 0.0,
            ..Default::default()
        },
    );

    let result = mcts.search(&SinglePathGame::default());

    assert_eq!(result.policy, vec![1.0]);
    assert_eq!(mcts.virtual_loss_total(), 0);
    assert_eq!(mcts.root_visits(), 4);
    assert_eq!(*calls.lock().unwrap(), vec![1, 1]);
}

#[test]
fn each_search_owns_a_fresh_tree_and_retains_only_capacity() {
    let mut mcts = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            simulations: 32,
            eps: 0.0,
            ..Default::default()
        },
    );

    let first = mcts.search(&Connect4::default());
    let first_nodes = mcts.node_count();
    let second = mcts.search(&Connect4::default());

    assert_eq!(first.policy, second.policy);
    assert_eq!(mcts.root_visits(), 32);
    assert_eq!(mcts.node_count(), first_nodes);
    assert_eq!(mcts.virtual_loss_total(), 0);
}

#[test]
fn batched_gumbel_clears_virtual_loss_and_keeps_valid_policy() {
    let game = Connect4::default();
    let mut mcts = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            variant: MctsVariant::Gumbel { sampled_actions: 4 },
            simulations: 32,
            leaf_batch_size: 8,
            eps: 0.0,
            ..Default::default()
        },
    );
    mcts.seed_rng(19);

    let result = mcts.search(&game);

    assert_eq!(mcts.virtual_loss_total(), 0);
    assert!((result.policy.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    assert!(mcts.gumbel_root_actions().contains(&result.best_action()));
}

#[test]
fn repetition_eval_cache_reuses_network_outputs() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let cache = Arc::new(EvalTable::new(16));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls.clone()),
        MctsConfig {
            simulations: 1,
            eps: 0.0,
            ..Default::default()
        },
    )
    .with_eval_cache(cache);

    let first = mcts.search_with_repetitions(&SinglePathGame::default(), |_| 0);
    let second = mcts.search_with_repetitions(&SinglePathGame::default(), |_| 0);

    assert_eq!(first.policy, vec![1.0]);
    assert_eq!(second.policy, vec![1.0]);
    assert_eq!(*calls.lock().unwrap(), vec![1, 1]);
}

#[test]
fn repetition_counts_the_root_when_a_branch_returns_to_it() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::uniform(calls.clone()),
        MctsConfig {
            simulations: 3,
            eps: 0.0,
            ..Default::default()
        },
    );

    mcts.search_with_repetitions(&CycleGame::default(), |hash| u8::from(hash == 1));

    // Root and its child require inference. The root had already occurred
    // once before this search, so the grandchild's root hash is the third
    // overall occurrence and must be backed up as a draw without inference.
    assert_eq!(*calls.lock().unwrap(), vec![1, 1]);
    assert_eq!(mcts.virtual_loss_total(), 0);
}

#[test]
fn repetition_search_populates_history_features_before_evaluation() {
    let encoded = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        FeatureRecordingEvaluator {
            encoded: encoded.clone(),
        },
        MctsConfig {
            simulations: 2,
            eps: 0.0,
            ..Default::default()
        },
    );

    mcts.search_with_repetitions(&FeatureRepetitionGame::default(), |hash| {
        u8::from(hash == 1)
    });

    let encoded = encoded.lock().unwrap();
    assert_eq!(encoded[0], 0.0);
    assert!(
        encoded.contains(&1.0),
        "the descendant's full-game repetition count was not encoded: {encoded:?}"
    );
}

#[test]
fn lossy_eval_table_overwrites_collisions() {
    let table = EvalTable::new(1);
    table.insert(
        1,
        CachedEvaluation {
            legal: vec![0],
            eval: Evaluation {
                logits: vec![1.0],
                value: 0.25,
            },
        },
    );
    assert_eq!(table.get(1).unwrap().eval.value, 0.25);

    table.insert(
        2,
        CachedEvaluation {
            legal: vec![0],
            eval: Evaluation {
                logits: vec![2.0],
                value: -0.5,
            },
        },
    );

    assert!(table.get(1).is_none());
    assert_eq!(table.get(2).unwrap().eval.value, -0.5);
}

#[test]
fn finds_immediate_win() {
    // X has three stones in column 0 and is to move.
    let game = play(&[0, 1, 0, 1, 0, 1]);
    let result = mcts(400).search(&game);
    assert_eq!(
        result.best_action(),
        0,
        "should pick the winning column: {:?}",
        result.policy
    );
}

#[test]
fn blocks_opponent_threat() {
    // X threatens to complete column 0; O to move must block it.
    let game = play(&[0, 6, 0, 5, 0]);
    let result = mcts(2000).search(&game);
    assert_eq!(
        result.best_action(),
        0,
        "should block column 0: {:?}",
        result.policy
    );
}

#[test]
fn policy_is_a_distribution() {
    let game = Connect4::default();
    let mut mcts = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            simulations: 200,
            eps: 0.0,
            ..Default::default()
        },
    );
    let result = mcts.search(&game);
    let sum: f32 = result.policy.iter().sum();
    assert!((sum - 1.0).abs() < 1e-4);
    assert!(result.policy.iter().all(|&p| p >= 0.0));
}

#[test]
fn gumbel_selected_action_stays_on_sampled_root_actions() {
    let game = Connect4::default();
    let mut mcts = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            variant: MctsVariant::Gumbel { sampled_actions: 3 },
            simulations: 48,
            eps: 0.0,
            ..Default::default()
        },
    );
    mcts.seed_rng(11);

    let result = mcts.search(&game);
    let sum: f32 = result.policy.iter().sum();
    let sampled = mcts.gumbel_root_actions();
    let selected_was_sampled = sampled.contains(&result.best_action());

    assert!((sum - 1.0).abs() < 1e-4);
    assert_eq!(sampled.len(), 3);
    assert!(selected_was_sampled, "{:?}", result.policy);
    assert!(result.policy.iter().all(|&p| p >= 0.0));
}

#[test]
#[should_panic(expected = "Gumbel sampled_actions must be positive")]
fn gumbel_rejects_zero_sampled_actions() {
    let _ = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            variant: MctsVariant::Gumbel { sampled_actions: 0 },
            simulations: 24,
            eps: 0.0,
            ..Default::default()
        },
    );
}

#[test]
fn gumbel_with_all_root_actions_finds_forced_terminal_win() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut mcts = Mcts::new(
        RecordingEvaluator::favor_action_zero(calls),
        MctsConfig {
            variant: MctsVariant::Gumbel { sampled_actions: 3 },
            simulations: 32,
            eps: 0.0,
            ..Default::default()
        },
    );
    mcts.seed_rng(3);

    let result = mcts.search(&ImmediateOutcomeGame::default());

    assert_eq!(result.best_action(), 0);
    assert!(result.policy[0] > result.policy[1], "{:?}", result.policy);
    assert!(result.policy[0] > result.policy[2], "{:?}", result.policy);
}

#[test]
#[should_panic(expected = "evaluator must return one result per input state")]
fn rejects_evaluator_result_count_mismatch() {
    let mut mcts = Mcts::new(
        EmptyResultEvaluator,
        MctsConfig {
            simulations: 1,
            eps: 0.0,
            ..Default::default()
        },
    );

    let _ = mcts.search(&ImmediateOutcomeGame::default());
}

#[test]
#[should_panic(expected = "evaluator logits must match the legal actions for each state")]
fn rejects_evaluator_logit_count_mismatch() {
    let mut mcts = Mcts::new(
        BadLogitEvaluator,
        MctsConfig {
            simulations: 1,
            eps: 0.0,
            ..Default::default()
        },
    );

    let _ = mcts.search(&ImmediateOutcomeGame::default());
}

#[test]
fn rejects_invalid_mcts_configs() {
    let invalid = [
        MctsConfig {
            simulations: 0,
            ..MctsConfig::default()
        },
        MctsConfig {
            leaf_batch_size: 0,
            ..MctsConfig::default()
        },
        MctsConfig {
            eps: f32::NAN,
            ..MctsConfig::default()
        },
        MctsConfig {
            alpha: 0.0,
            ..MctsConfig::default()
        },
        MctsConfig {
            variant: MctsVariant::Gumbel { sampled_actions: 0 },
            ..MctsConfig::default()
        },
    ];

    for cfg in invalid {
        assert!(cfg.validate().is_err(), "accepted invalid config: {cfg:?}");
    }
}

#[test]
#[should_panic(expected = "MCTS simulations must be positive")]
fn rejects_zero_simulations_at_runtime() {
    let mut mcts = mcts(1);
    mcts.set_simulations(0);
}

#[test]
fn gumbel_profile_updates_simulations_and_root_candidates_together() {
    let mut mcts = Mcts::new(
        UniformEvaluator,
        MctsConfig {
            simulations: 128,
            variant: MctsVariant::Gumbel {
                sampled_actions: 16,
            },
            ..MctsConfig::default()
        },
    );
    mcts.set_gumbel_profile(GumbelSearchProfile::new(64, 8));
    assert_eq!(mcts.config().simulations, 64);
    assert_eq!(
        mcts.config().variant,
        MctsVariant::Gumbel { sampled_actions: 8 }
    );
}

/// Dependency-free throughput smoke harness. It deliberately has no timing
/// assertion: host load and build profiles make wall-clock thresholds flaky.
/// Run with:
/// `cargo test -p algorithms --release mcts_release_throughput -- --ignored --nocapture`
#[test]
#[ignore = "manual release-mode MCTS throughput harness"]
fn mcts_release_throughput() {
    const SEARCHES: usize = 200;
    const SIMULATIONS: usize = 256;

    let calls = Arc::new(AtomicUsize::new(0));
    let mut mcts = Mcts::new(
        CountingEvaluator {
            calls: Arc::clone(&calls),
        },
        MctsConfig {
            simulations: SIMULATIONS,
            leaf_batch_size: 1,
            eps: 0.0,
            ..Default::default()
        },
    );
    let game = Connect4::default();

    // Warm reusable tree/evaluation buffers before measuring.
    let _ = mcts.search(&game);
    calls.store(0, Ordering::Relaxed);
    let started = Instant::now();
    for _ in 0..SEARCHES {
        std::hint::black_box(mcts.search(std::hint::black_box(&game)));
    }
    let elapsed = started.elapsed();
    let searches_per_second = SEARCHES as f64 / elapsed.as_secs_f64();
    let simulations_per_second = (SEARCHES * SIMULATIONS) as f64 / elapsed.as_secs_f64();

    eprintln!(
        "mcts: {SEARCHES} searches x {SIMULATIONS} simulations in {elapsed:?}; \
         {searches_per_second:.1} searches/s, {simulations_per_second:.0} simulations/s, \
         {} evaluator calls",
        calls.load(Ordering::Relaxed)
    );
}
