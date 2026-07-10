use algorithms::alphazero::{Batcher, Mcts, MctsConfig, RunConfig};
use anyhow::{bail, Context, Result};
use chess::{Board, BoardStatus, ChessMove, MoveGen, Piece};
use clap::Parser;
use engine_core::game::Game;
use games::chess::decode_move;
use games::ChessGame;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::Serialize;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tch::Device;

#[derive(Parser)]
#[command(about = "Run a small, balanced Stockfish match for one chess checkpoint")]
struct Args {
    #[arg(long)]
    run_dir: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long, default_value = "stockfish")]
    stockfish: PathBuf,
    #[arg(long, default_value_t = 1400)]
    stockfish_elo: u32,
    #[arg(long, default_value_t = 4)]
    games: usize,
    /// Checkpoint from the prior evaluation interval for a direct relative match.
    #[arg(long)]
    baseline: Option<PathBuf>,
    #[arg(long, default_value_t = 4)]
    baseline_games: usize,
    #[arg(long, default_value_t = 400)]
    simulations: usize,
    #[arg(long, default_value_t = 200)]
    stockfish_movetime_ms: u64,
    #[arg(long, default_value_t = 1)]
    stockfish_threads: u32,
    #[arg(long, default_value_t = 4)]
    opening_plies: usize,
    #[arg(long, default_value_t = 512)]
    max_moves: usize,
    #[arg(long, default_value = "cpu")]
    device: String,
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Serialize)]
struct ResultRecord {
    checkpoint: String,
    stockfish_elo: u32,
    games: usize,
    model_wins: usize,
    draws: usize,
    model_losses: usize,
    score: f64,
    score_pct: f64,
    rough_elo_delta: f64,
    simulations: usize,
    stockfish_movetime_ms: u64,
    opening_plies: usize,
    seed: u64,
    pgn_dir: String,
    baseline: Option<String>,
    baseline_games: Option<usize>,
    baseline_wins: Option<usize>,
    baseline_draws: Option<usize>,
    baseline_losses: Option<usize>,
    baseline_score_pct: Option<f64>,
    baseline_rough_elo_delta: Option<f64>,
    time: f64,
}

struct Stockfish {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Stockfish {
    fn start(path: &Path, elo: u32, threads: u32) -> Result<Self> {
        let mut child = Command::new(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("starting Stockfish at {}", path.display()))?;
        let input = BufWriter::new(child.stdin.take().context("Stockfish stdin unavailable")?);
        let output = BufReader::new(
            child
                .stdout
                .take()
                .context("Stockfish stdout unavailable")?,
        );
        let mut sf = Self {
            child,
            input,
            output,
        };
        sf.send("uci")?;
        sf.wait_for("uciok")?;
        sf.send("setoption name UCI_LimitStrength value true")?;
        sf.send(&format!("setoption name UCI_Elo value {elo}"))?;
        sf.send(&format!("setoption name Threads value {threads}"))?;
        sf.ready()?;
        Ok(sf)
    }

    fn send(&mut self, line: &str) -> Result<()> {
        writeln!(self.input, "{line}")?;
        self.input.flush()?;
        Ok(())
    }

    fn wait_for(&mut self, expected: &str) -> Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            anyhow::ensure!(
                self.output.read_line(&mut line)? != 0,
                "Stockfish exited unexpectedly"
            );
            if line.trim() == expected {
                return Ok(());
            }
        }
    }

    fn ready(&mut self) -> Result<()> {
        self.send("isready")?;
        self.wait_for("readyok")
    }

    fn best_move(&mut self, moves: &[String], movetime_ms: u64) -> Result<String> {
        self.send("ucinewgame")?;
        self.ready()?;
        let position = if moves.is_empty() {
            "position startpos".to_owned()
        }
        else {
            format!("position startpos moves {}", moves.join(" "))
        };
        self.send(&position)?;
        self.send(&format!("go movetime {movetime_ms}"))?;
        let mut line = String::new();
        loop {
            line.clear();
            anyhow::ensure!(
                self.output.read_line(&mut line)? != 0,
                "Stockfish exited unexpectedly"
            );
            if let Some(mv) = line.trim().strip_prefix("bestmove ") {
                return mv
                    .split_whitespace()
                    .next()
                    .context("Stockfish returned no move")
                    .map(str::to_owned);
            }
        }
    }
}

impl Drop for Stockfish {
    fn drop(&mut self) {
        let _ = self.send("quit");
        let _ = self.child.wait();
    }
}

fn san(board: &Board, action: u32) -> String {
    let mv = decode_move(action);
    let piece = board
        .piece_on(mv.get_source())
        .expect("legal move has a piece");
    let capture = board.piece_on(mv.get_dest()).is_some()
        || (piece == Piece::Pawn && mv.get_source().get_file() != mv.get_dest().get_file());
    if piece == Piece::King
        && (mv.get_source().get_file().to_index() as i32
            - mv.get_dest().get_file().to_index() as i32)
            .abs()
            == 2
    {
        let next = board.make_move_new(mv);
        let castle = if mv.get_dest().get_file().to_index() > mv.get_source().get_file().to_index()
        {
            "O-O"
        }
        else {
            "O-O-O"
        };
        return format!("{castle}{}", suffix(&next));
    }
    let mut out = String::new();
    if piece != Piece::Pawn {
        out.push(match piece {
            Piece::Knight => 'N',
            Piece::Bishop => 'B',
            Piece::Rook => 'R',
            Piece::Queen => 'Q',
            Piece::King => 'K',
            Piece::Pawn => unreachable!(),
        });
        let same_target: Vec<ChessMove> = MoveGen::new_legal(board)
            .filter(|other| {
                other.get_dest() == mv.get_dest()
                    && other.get_source() != mv.get_source()
                    && board.piece_on(other.get_source()) == Some(piece)
            })
            .collect();
        if !same_target.is_empty() {
            let file = mv.get_source().get_file();
            let rank = mv.get_source().get_rank();
            if same_target
                .iter()
                .all(|other| other.get_source().get_file() != file)
            {
                out.push(char::from(b'a' + file.to_index() as u8));
            }
            else if same_target
                .iter()
                .all(|other| other.get_source().get_rank() != rank)
            {
                out.push(char::from(b'1' + rank.to_index() as u8));
            }
            else {
                out.push_str(&mv.get_source().to_string());
            }
        }
    }
    else if capture {
        out.push(char::from(
            b'a' + mv.get_source().get_file().to_index() as u8,
        ));
    }
    if capture {
        out.push('x');
    }
    out.push_str(&mv.get_dest().to_string());
    if let Some(promotion) = mv.get_promotion() {
        out.push('=');
        out.push(match promotion {
            Piece::Queen => 'Q',
            Piece::Rook => 'R',
            Piece::Bishop => 'B',
            Piece::Knight => 'N',
            _ => unreachable!(),
        });
    }
    out.push_str(suffix(&board.make_move_new(mv)));
    out
}

fn suffix(board: &Board) -> &'static str {
    if board.status() == BoardStatus::Checkmate {
        "#"
    }
    else if board.checkers().popcnt() > 0 {
        "+"
    }
    else {
        ""
    }
}

fn apply_opening(
    game: &mut ChessGame,
    board: &mut Board,
    moves: &mut Vec<String>,
    plies: usize,
    rng: &mut StdRng,
) {
    for _ in 0..plies {
        if game.is_terminal() {
            break;
        }
        let legal: Vec<_> = game.legal_actions().collect();
        let action = legal[rng.random_range(0..legal.len())];
        moves.push(game.format_action(action));
        *board = board.make_move_new(decode_move(action));
        game.step(action);
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    anyhow::ensure!(
        args.games > 0 && args.games.is_multiple_of(2),
        "--games must be a positive even number"
    );
    let device = match args.device.as_str() {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda(0),
        "auto" => Device::cuda_if_available(),
        other => bail!("unknown --device {other}; use cpu, cuda, or auto"),
    };
    let config: RunConfig =
        serde_json::from_str(&fs::read_to_string(args.run_dir.join("config.json"))?)?;
    anyhow::ensure!(
        config.game == "chess",
        "{} is not a chess run",
        args.run_dir.display()
    );
    let batcher = Batcher::new(
        &config.net,
        &args.checkpoint,
        device,
        1,
        Duration::from_millis(2),
    )?;
    let mut model = Mcts::new(
        batcher.client(),
        MctsConfig {
            simulations: args.simulations,
            eps: 0.0,
            ..Default::default()
        },
    );
    let mut stockfish =
        Stockfish::start(&args.stockfish, args.stockfish_elo, args.stockfish_threads)?;
    let name = args
        .checkpoint
        .file_stem()
        .context("checkpoint needs a filename")?
        .to_string_lossy()
        .to_string();
    let out_dir = args.run_dir.join("evaluations");
    let pgn_dir = out_dir.join("pgn").join(&name).join("stockfish");
    fs::create_dir_all(&pgn_dir)?;
    let seed = args.seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    });
    let mut rng = StdRng::seed_from_u64(seed);
    let (mut wins, mut draws, mut losses) = (0usize, 0usize, 0usize);
    for game_idx in 0..args.games {
        let model_white = game_idx.is_multiple_of(2);
        let mut game = ChessGame::default();
        let mut board = Board::default();
        let mut uci_moves = Vec::new();
        apply_opening(
            &mut game,
            &mut board,
            &mut uci_moves,
            args.opening_plies,
            &mut rng,
        );
        // Rebuild standard SAN movetext from the complete UCI sequence after the game.
        let mut ply = uci_moves.len();
        while !game.is_terminal() && ply < args.max_moves {
            let model_turn = (ply.is_multiple_of(2)) == model_white;
            let action = if model_turn {
                model
                    .search_with_mode(&game, engine_core::agent::PolicyMode::Deterministic)
                    .best_action()
            }
            else {
                let mv = stockfish.best_move(&uci_moves, args.stockfish_movetime_ms)?;
                game.parse_move(&mv)
                    .with_context(|| format!("Stockfish returned illegal move {mv}"))?
            };
            uci_moves.push(game.format_action(action));
            game.step(action);
            ply += 1;
        }
        let result = if !game.is_terminal() || game.reward() == 0.0 {
            draws += 1;
            "1/2-1/2"
        }
        else if (ply - 1).is_multiple_of(2) == model_white {
            wins += 1;
            if model_white {
                "1-0"
            }
            else {
                "0-1"
            }
        }
        else {
            losses += 1;
            if model_white {
                "0-1"
            }
            else {
                "1-0"
            }
        };
        let mut replay_board = Board::default();
        let mut movetext = String::new();
        for (i, uci) in uci_moves.iter().enumerate() {
            let action = ChessGame::default()
                .parse_move(uci)
                .or_else(|| {
                    let mv = uci.parse().ok()?;
                    Some(games::chess::encode_move(mv))
                })
                .expect("valid UCI move");
            if i.is_multiple_of(2) {
                movetext.push_str(&format!("{}. ", i / 2 + 1));
            }
            movetext.push_str(&san(&replay_board, action));
            movetext.push(' ');
            replay_board = replay_board.make_move_new(decode_move(action));
        }
        let pgn = format!("[Event \"engine-zoo checkpoint evaluation\"]\n[Round \"{}\"]\n[White \"{}\"]\n[Black \"{}\"]\n[Result \"{}\"]\n[StockfishElo \"{}\"]\n\n{}{}\n", game_idx + 1, if model_white { &name } else { "Stockfish" }, if model_white { "Stockfish" } else { &name }, result, args.stockfish_elo, movetext, result);
        fs::write(pgn_dir.join(format!("game_{:02}.pgn", game_idx + 1)), pgn)?;
    }
    let score = wins as f64 + draws as f64 * 0.5;
    let smoothed = (score + 0.5) / (args.games as f64 + 1.0);
    let baseline_result = if let Some(path) = args.baseline.as_ref() {
        anyhow::ensure!(
            args.baseline_games > 0 && args.baseline_games.is_multiple_of(2),
            "--baseline-games must be a positive even number"
        );
        let baseline_batcher =
            Batcher::new(&config.net, path, device, 1, Duration::from_millis(2))?;
        let mut baseline = Mcts::new(
            baseline_batcher.client(),
            MctsConfig {
                simulations: args.simulations,
                eps: 0.0,
                ..Default::default()
            },
        );
        let (mut baseline_wins, mut baseline_draws, mut baseline_losses) = (0usize, 0usize, 0usize);
        for game_idx in 0..args.baseline_games {
            let model_white = game_idx.is_multiple_of(2);
            let mut game = ChessGame::default();
            let mut board = Board::default();
            let mut uci_moves = Vec::new();
            apply_opening(
                &mut game,
                &mut board,
                &mut uci_moves,
                args.opening_plies,
                &mut rng,
            );
            let mut ply = uci_moves.len();
            while !game.is_terminal() && ply < args.max_moves {
                let action = if ply.is_multiple_of(2) == model_white {
                    model
                        .search_with_mode(&game, engine_core::agent::PolicyMode::Deterministic)
                        .best_action()
                }
                else {
                    baseline
                        .search_with_mode(&game, engine_core::agent::PolicyMode::Deterministic)
                        .best_action()
                };
                game.step(action);
                ply += 1;
            }
            if !game.is_terminal() || game.reward() == 0.0 {
                baseline_draws += 1;
            }
            else if (ply - 1).is_multiple_of(2) == model_white {
                baseline_wins += 1;
            }
            else {
                baseline_losses += 1;
            }
        }
        let baseline_score = baseline_wins as f64 + baseline_draws as f64 * 0.5;
        let baseline_smoothed = (baseline_score + 0.5) / (args.baseline_games as f64 + 1.0);
        Some((
            path.display().to_string(),
            baseline_wins,
            baseline_draws,
            baseline_losses,
            baseline_score * 100.0 / args.baseline_games as f64,
            400.0 * (baseline_smoothed / (1.0 - baseline_smoothed)).log10(),
        ))
    }
    else {
        None
    };
    let record = ResultRecord {
        checkpoint: args.checkpoint.display().to_string(),
        stockfish_elo: args.stockfish_elo,
        games: args.games,
        model_wins: wins,
        draws,
        model_losses: losses,
        score,
        score_pct: score * 100.0 / args.games as f64,
        rough_elo_delta: 400.0 * (smoothed / (1.0 - smoothed)).log10(),
        simulations: args.simulations,
        stockfish_movetime_ms: args.stockfish_movetime_ms,
        opening_plies: args.opening_plies,
        seed,
        pgn_dir: pgn_dir.display().to_string(),
        baseline: baseline_result.as_ref().map(|x| x.0.clone()),
        baseline_games: baseline_result.as_ref().map(|_| args.baseline_games),
        baseline_wins: baseline_result.as_ref().map(|x| x.1),
        baseline_draws: baseline_result.as_ref().map(|x| x.2),
        baseline_losses: baseline_result.as_ref().map(|x| x.3),
        baseline_score_pct: baseline_result.as_ref().map(|x| x.4),
        baseline_rough_elo_delta: baseline_result.as_ref().map(|x| x.5),
        time: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64(),
    };
    let line = serde_json::to_string(&record)?;
    fs::create_dir_all(&out_dir)?;
    let results_path = out_dir.join("ratings.jsonl");
    let mut results = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&results_path)?;
    writeln!(results, "{line}")?;
    fs::write(out_dir.join(format!("{name}.json")), format!("{line}\n"))?;
    println!("evaluation: {line}");
    Ok(())
}
