use alphazero::representation::{ChessAzRepresentation, ChessClassicRepresentation};
use alphazero::ChessRepetitionRules;
use alphazero::{Batcher, BatcherConfig, ChessHistory, GameKind, RepresentedEvaluator};
use anyhow::{Context, Result};
use engine_core::notation::GameNotation;
use games::chess::ChessUciNotation;
use games::ChessGame;
use search::NoExtraRules;
use search::{Mcts, SearchBudget, SearchConfig, SearchRequest};
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

impl PuzzleSummary {
    fn record(&mut self, correct: bool, category: Option<&str>, tier: Option<&str>) {
        self.total += 1;
        self.correct += usize::from(correct);

        if let Some(category) = category {
            self.category
                .entry(category.to_owned())
                .or_default()
                .add(correct);
        }

        if let Some(tier) = tier {
            self.tier.entry(tier.to_owned()).or_default().add(correct);
        }
    }

    fn finish(&mut self) {
        self.accuracy = if self.total == 0 {
            0.0
        }
        else {
            self.correct as f64 / self.total as f64
        };
    }
}

impl Default for PuzzleSummary {
    fn default() -> Self {
        Self {
            record_type: "summary",
            total: 0,
            correct: 0,
            accuracy: 0.0,
            category: BTreeMap::new(),
            tier: BTreeMap::new(),
        }
    }
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
    let mut summary = PuzzleSummary::default();

    for puzzle in puzzles {
        let result = evaluate_puzzle(puzzle, &mut candidate_move)?;

        summary.record(
            result.correct,
            result.category.as_deref(),
            result.tier.as_deref(),
        );
        results.push(result);
    }

    summary.finish();

    Ok((results, summary))
}

fn evaluate_puzzle(
    puzzle: &Puzzle,
    candidate_move: &mut impl FnMut(&Puzzle) -> Result<String>,
) -> Result<PuzzleResult> {
    let game = ChessGame::from_fen(&puzzle.fen)
        .with_context(|| format!("invalid FEN for puzzle {}", puzzle.id))?;

    validate_accepted_moves(puzzle, &game)?;

    let move_played = candidate_move(puzzle)?;
    validate_candidate_move(puzzle, &game, &move_played)?;

    let correct = puzzle.accepted_moves.iter().any(|mv| mv == &move_played);

    Ok(PuzzleResult {
        record_type: "puzzle",
        id: puzzle.id.clone(),
        candidate_move: move_played,
        accepted_moves: puzzle.accepted_moves.clone(),
        correct,
        category: puzzle.category.clone(),
        tier: puzzle.tier.clone(),
    })
}

fn validate_accepted_moves(puzzle: &Puzzle, game: &ChessGame) -> Result<()> {
    anyhow::ensure!(
        !puzzle.accepted_moves.is_empty(),
        "puzzle {} has no accepted moves",
        puzzle.id
    );

    for accepted in &puzzle.accepted_moves {
        let action = ChessUciNotation
            .parse_move(&game.position(), accepted)
            .with_context(|| {
                format!(
                    "puzzle {} has illegal accepted UCI move {accepted}",
                    puzzle.id
                )
            })?;
        anyhow::ensure!(
            ChessUciNotation.format_move(&game.position(), action) == *accepted,
            "puzzle {} has non-canonical accepted UCI move {accepted}",
            puzzle.id
        );
    }

    Ok(())
}

fn validate_candidate_move(puzzle: &Puzzle, game: &ChessGame, move_played: &str) -> Result<()> {
    let action = ChessUciNotation
        .parse_move(&game.position(), move_played)
        .with_context(|| {
            format!(
                "candidate returned illegal UCI move {move_played} for puzzle {}",
                puzzle.id
            )
        })?;
    anyhow::ensure!(
        ChessUciNotation.format_move(&game.position(), action) == move_played,
        "candidate returned non-canonical UCI move {move_played} for puzzle {}",
        puzzle.id
    );

    Ok(())
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
    mcts_config: SearchConfig,
    mcts_budget: SearchBudget,
) -> Result<(Vec<PuzzleResult>, PuzzleSummary)> {
    let (_, config, _) = alphazero::RunDir::open_or_create(run_dir, || {
        panic!("no experiment found at {}", run_dir.display())
    })?;
    anyhow::ensure!(
        config.model.game() == GameKind::Chess,
        "{} is not a chess run",
        run_dir.display()
    );
    let batcher =
        Batcher::new_with_model(config.model.clone(), checkpoint, device, batcher_config())?;
    match config.model.chess_history() {
        Some(ChessHistory::One) => {
            evaluate_chess_puzzles::<1>(puzzles, batcher.client(), mcts_config, mcts_budget)
        }
        Some(ChessHistory::Four) => {
            evaluate_chess_puzzles::<4>(puzzles, batcher.client(), mcts_config, mcts_budget)
        }
        Some(ChessHistory::Eight) => {
            evaluate_chess_puzzles::<8>(puzzles, batcher.client(), mcts_config, mcts_budget)
        }
        None if config.model.is_chess_classic() => {
            evaluate_classic_chess_puzzles(puzzles, batcher.client(), mcts_config, mcts_budget)
        }
        None => unreachable!("validated chess run"),
    }
}

fn evaluate_classic_chess_puzzles(
    puzzles: &[Puzzle],
    evaluator: alphazero::BatcherClient,
    mcts_config: SearchConfig,
    mcts_budget: SearchBudget,
) -> Result<(Vec<PuzzleResult>, PuzzleSummary)> {
    let mut mcts = Mcts::new(
        RepresentedEvaluator::new(ChessClassicRepresentation, evaluator),
        mcts_config,
        NoExtraRules,
    );
    evaluate_moves(puzzles, |puzzle| {
        let game = ChessGame::from_fen(&puzzle.fen)?;
        let mv = mcts
            .search(&game.position(), (), deterministic_request(mcts_budget))?
            .best_move();
        Ok(games::chess::ChessUciNotation.format_move(&game.position(), mv))
    })
}

fn evaluate_chess_puzzles<const HISTORY: usize>(
    puzzles: &[Puzzle],
    evaluator: alphazero::BatcherClient,
    mcts_config: SearchConfig,
    mcts_budget: SearchBudget,
) -> Result<(Vec<PuzzleResult>, PuzzleSummary)> {
    let mut mcts = Mcts::new(
        RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, evaluator),
        mcts_config,
        ChessRepetitionRules,
    );
    evaluate_moves(puzzles, |puzzle| {
        let game = ChessGame::from_fen(&puzzle.fen)?;
        let state = alphazero::representation::ChessAzState::from_game(&game);
        let repetition_context = game.repetition_context();
        let mv = mcts
            .search(
                &state,
                repetition_context,
                deterministic_request(mcts_budget),
            )?
            .best_move();
        Ok(games::chess::ChessUciNotation.format_move(&game.position(), mv))
    })
}

fn deterministic_request(budget: SearchBudget) -> SearchRequest {
    SearchRequest {
        mode: engine_core::agent::PolicyMode::Deterministic,
        budget,
    }
}

fn batcher_config() -> BatcherConfig {
    BatcherConfig {
        preferred_batch_size: 1,
        max_batch_size: 128,
        max_wait: Duration::from_millis(2),
        max_queued_states: 1024,
    }
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
    fn chess_run_exposes_its_history_through_the_model_spec() {
        let model = alphazero::ModelSpec::chess_se(
            alphazero::ChessHistory::Four,
            alphazero::ValueHeadSpec::Wdl { hidden: 128 },
        );
        assert_eq!(model.chess_history(), Some(alphazero::ChessHistory::Four));
    }
}
