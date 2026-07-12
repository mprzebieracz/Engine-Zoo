//! Stockfish protocol support for the legacy evaluator.

use anyhow::{Context, Result};
use engine_core::game::Game;
use games::ChessGame;
use rand::{rngs::StdRng, Rng};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub(super) struct Stockfish {
    child: Child,
    input: BufWriter<ChildStdin>,
    output: BufReader<ChildStdout>,
}

impl Stockfish {
    pub(super) fn start(path: &Path, elo: u32, threads: u32) -> Result<Self> {
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

    pub(super) fn best_move(&mut self, moves: &[String], movetime_ms: u64) -> Result<String> {
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

pub(super) fn apply_opening(
    game: &mut ChessGame,
    mut notation: Option<(&mut Vec<String>, &mut Vec<String>)>,
    plies: usize,
    rng: &mut StdRng,
) -> usize {
    let mut applied = 0;
    for _ in 0..plies {
        if game.is_terminal() {
            break;
        }
        let legal: Vec<_> = game.legal_actions().collect();
        let action = legal[rng.random_range(0..legal.len())];
        if let Some((uci_moves, san_moves)) = &mut notation {
            san_moves.push(game.san_for_action(action));
            uci_moves.push(game.format_action(action));
        }
        game.step(action);
        applied += 1;
    }
    applied
}
