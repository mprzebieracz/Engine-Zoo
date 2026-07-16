use algorithms::alphazero::{Batcher, Mcts, MctsConfig, RunArchitecture};
use anyhow::{Context, Result};
use engine_core::agent::PolicyMode;
use engine_core::game::Game;
use engine_core::notation::GameNotation;
use games::{decode_v2_action, ChessGame};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Duration;
use tch::Device;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Puzzle {
    pub id: String,
    pub fen: String,
    pub accepted_moves: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PuzzleResult {
    pub record_type: &'static str,
    pub id: String,
    pub candidate_move: String,
    pub accepted_moves: Vec<String>,
    pub correct: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub category: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct AggregateStats {
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
}

impl AggregateStats {
    fn add(&mut self, correct: bool) {
        self.total += 1;
        self.correct += usize::from(correct);
        self.accuracy = self.correct as f64 / self.total as f64;
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct PuzzleSummary {
    pub record_type: &'static str,
    pub total: usize,
    pub correct: usize,
    pub accuracy: f64,
    pub category: BTreeMap<String, AggregateStats>,
    pub tier: BTreeMap<String, AggregateStats>,
}

pub fn read_puzzles(reader: impl BufRead) -> Result<Vec<Puzzle>> {
    let mut puzzles = Vec::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading puzzle JSONL line {}", line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let puzzle = serde_json::from_str(&line)
            .with_context(|| format!("parsing puzzle JSONL line {}", line_no + 1))?;
        puzzles.push(puzzle);
    }
    Ok(puzzles)
}

pub fn evaluate_moves(
    puzzles: &[Puzzle],
    mut candidate_move: impl FnMut(&Puzzle) -> Result<String>,
) -> Result<(Vec<PuzzleResult>, PuzzleSummary)> {
    let mut results = Vec::with_capacity(puzzles.len());
    let mut summary = PuzzleSummary {
        record_type: "summary",
        total: 0,
        correct: 0,
        accuracy: 0.0,
        category: BTreeMap::new(),
        tier: BTreeMap::new(),
    };

    for puzzle in puzzles {
        let game = ChessGame::from_fen(&puzzle.fen)
            .with_context(|| format!("invalid FEN for puzzle {}", puzzle.id))?;
        for accepted in &puzzle.accepted_moves {
            let action = game.parse_move(accepted).with_context(|| {
                format!(
                    "puzzle {} has illegal accepted UCI move {accepted}",
                    puzzle.id
                )
            })?;
            anyhow::ensure!(
                game.format_action(action) == *accepted,
                "puzzle {} has non-canonical accepted UCI move {accepted}",
                puzzle.id
            );
        }
        anyhow::ensure!(
            !puzzle.accepted_moves.is_empty(),
            "puzzle {} has no accepted moves",
            puzzle.id
        );
        let move_played = candidate_move(puzzle)?;
        let action = game.parse_move(&move_played).with_context(|| {
            format!(
                "candidate returned illegal UCI move {move_played} for puzzle {}",
                puzzle.id
            )
        })?;
        anyhow::ensure!(
            game.format_action(action) == move_played,
            "candidate returned non-canonical UCI move {move_played} for puzzle {}",
            puzzle.id
        );
        let correct = puzzle.accepted_moves.iter().any(|mv| mv == &move_played);
        results.push(PuzzleResult {
            record_type: "puzzle",
            id: puzzle.id.clone(),
            candidate_move: move_played,
            accepted_moves: puzzle.accepted_moves.clone(),
            correct,
            category: puzzle.category.clone(),
            tier: puzzle.tier.clone(),
        });
        summary.total += 1;
        summary.correct += usize::from(correct);
        if let Some(category) = &puzzle.category {
            summary
                .category
                .entry(category.clone())
                .or_default()
                .add(correct);
        }
        if let Some(tier) = &puzzle.tier {
            summary.tier.entry(tier.clone()).or_default().add(correct);
        }
    }
    summary.accuracy = if summary.total == 0 {
        0.0
    }
    else {
        summary.correct as f64 / summary.total as f64
    };
    Ok((results, summary))
}

pub fn write_jsonl(
    mut writer: impl Write,
    results: &[PuzzleResult],
    summary: &PuzzleSummary,
) -> Result<()> {
    for result in results {
        serde_json::to_writer(&mut writer, result)?;
        writeln!(writer)?;
    }
    serde_json::to_writer(&mut writer, summary)?;
    writeln!(writer)?;
    Ok(())
}

pub fn evaluate_checkpoint(
    puzzles: &[Puzzle],
    run_dir: &Path,
    checkpoint: &Path,
    device: Device,
    mcts_config: MctsConfig,
) -> Result<(Vec<PuzzleResult>, PuzzleSummary)> {
    let config: algorithms::alphazero::RunConfig = serde_json::from_str(
        &std::fs::read_to_string(run_dir.join("config.json"))
            .with_context(|| format!("reading {}", run_dir.join("config.json").display()))?,
    )
    .with_context(|| format!("parsing {}/config.json", run_dir.display()))?;
    anyhow::ensure!(
        config.game == "chess",
        "{} is not a chess run",
        run_dir.display()
    );
    let network = config.network_config();
    let batcher =
        Batcher::new_with_network(&network, checkpoint, device, 1, Duration::from_millis(2))?;
    match config.architecture {
        RunArchitecture::Legacy => evaluate_legacy_puzzles(puzzles, batcher.client(), mcts_config),
        RunArchitecture::ChessAzV2(v2) => match v2.history {
            1 => evaluate_v2_puzzles::<1>(puzzles, batcher.client(), mcts_config),
            4 => evaluate_v2_puzzles::<4>(puzzles, batcher.client(), mcts_config),
            8 => evaluate_v2_puzzles::<8>(puzzles, batcher.client(), mcts_config),
            history => anyhow::bail!("unsupported chess-az-v2 history length {history}"),
        },
    }
}

fn evaluate_legacy_puzzles(
    puzzles: &[Puzzle],
    evaluator: algorithms::alphazero::BatcherClient,
    mcts_config: MctsConfig,
) -> Result<(Vec<PuzzleResult>, PuzzleSummary)> {
    let mut mcts = Mcts::new(evaluator, mcts_config);
    evaluate_moves(puzzles, |puzzle| {
        let game = ChessGame::from_fen(&puzzle.fen)?;
        let action = mcts
            .search_with_mode(&game, PolicyMode::Deterministic)
            .best_action();
        Ok(game.format_action(action))
    })
}

fn evaluate_v2_puzzles<const HISTORY: usize>(
    puzzles: &[Puzzle],
    evaluator: algorithms::alphazero::BatcherClient,
    mcts_config: MctsConfig,
) -> Result<(Vec<PuzzleResult>, PuzzleSummary)> {
    let mut mcts = Mcts::new(evaluator, mcts_config);
    evaluate_moves(puzzles, |puzzle| {
        let game = ChessGame::from_fen(&puzzle.fen)?;
        let state = game.history_state::<HISTORY>();
        let repetition_context = game.repetition_context();
        let action = mcts
            .search_with_repetitions_mode(
                &state,
                |hash| repetition_context.occurrences_before_root(hash),
                PolicyMode::Deterministic,
            )
            .best_action();
        let mv = decode_v2_action(game.board(), action).context("decoding puzzle v2 action")?;
        Ok(games::chess::ChessUciNotation.format_move(&game.position(), mv))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn puzzle(id: &str, category: Option<&str>, tier: Option<&str>) -> Puzzle {
        Puzzle {
            id: id.into(),
            fen: "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1".into(),
            accepted_moves: vec!["e1g1".into()],
            category: category.map(str::to_owned),
            tier: tier.map(str::to_owned),
        }
    }

    #[test]
    fn reads_jsonl_and_preserves_optional_metadata() {
        let input =
            br#"{"id":"smoke","fen":"8/8/8/8/8/8/8/K6k w - - 0 1","accepted_moves":["a1b1"]}
"#;
        let puzzles = read_puzzles(Cursor::new(input)).unwrap();
        assert_eq!(puzzles.len(), 1);
        assert_eq!(puzzles[0].category, None);
    }

    #[test]
    fn exact_move_matching_and_group_stats_are_deterministic() {
        let puzzles = vec![
            puzzle("b", Some("tactic"), Some("hard")),
            puzzle("a", Some("tactic"), None),
        ];
        let (results, summary) = evaluate_moves(&puzzles, |p| {
            Ok(if p.id == "b" { "e1c1" } else { "e1g1" }.into())
        })
        .unwrap();
        assert!(!results[0].correct);
        assert!(results[1].correct);
        assert_eq!(summary.accuracy, 0.5);
        assert_eq!(summary.category["tactic"].correct, 1);
        assert_eq!(summary.tier["hard"].correct, 0);
    }

    #[test]
    fn rejects_illegal_accepted_moves() {
        let mut p = puzzle("bad", None, None);
        p.accepted_moves = vec!["a1a1".into()];
        assert!(evaluate_moves(&[p], |_| Ok("a1b1".into())).is_err());
    }

    #[test]
    fn checked_in_tactical_suite_has_legal_canonical_moves() {
        let puzzles = read_puzzles(Cursor::new(include_str!(
            "../suites/chess/puzzles-tactics-v1.jsonl"
        )))
        .unwrap();
        assert!(puzzles.len() >= 16);
        assert!(
            evaluate_moves(&puzzles, |puzzle| { Ok(puzzle.accepted_moves[0].clone()) }).is_ok()
        );
    }

    #[test]
    fn empty_suite_has_a_finite_empty_summary_and_valid_jsonl() {
        let (results, summary) = evaluate_moves(&[], |_| unreachable!()).unwrap();
        assert_eq!(summary.total, 0);
        assert_eq!(summary.accuracy, 0.0);
        let mut output = Vec::new();
        write_jsonl(&mut output, &results, &summary).unwrap();
        let records: Vec<serde_json::Value> = String::from_utf8(output)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0]["record_type"], "summary");
    }

    #[test]
    fn read_errors_report_the_jsonl_line_number() {
        let error = read_puzzles(Cursor::new("\nnot-json\n")).unwrap_err();
        assert!(error.to_string().contains("line 2"));
    }

    #[test]
    fn v2_runs_select_the_explicit_v2_network_format() {
        let legacy = algorithms::alphazero::RunConfig {
            game: "chess".into(),
            net: algorithms::alphazero::NetConfig {
                input_channels: 1,
                height: 1,
                width: 1,
                num_res_blocks: 1,
                num_filters: 1,
                action_size: 1,
            },
            architecture: RunArchitecture::Legacy,
        };
        assert!(matches!(
            legacy.network_config(),
            algorithms::alphazero::NetworkConfig::Legacy(_)
        ));
        let v2 = algorithms::alphazero::RunConfig {
            architecture: RunArchitecture::ChessAzV2(Default::default()),
            ..legacy
        };
        assert!(matches!(
            v2.network_config(),
            algorithms::alphazero::NetworkConfig::ChessAzV2(_)
        ));
    }
}
