//! Shared persistence for Fastchess-backed match suites.

use crate::fastchess::{self, FastchessCommand, FastchessRun};
use crate::report::{write_report, EvaluationReport, EvaluationSpec, Score};
use anyhow::{bail, Context, Result};
use std::fs;
use std::path::Path;

pub fn run_match(
    output_dir: &Path,
    mut spec: EvaluationSpec,
    command: FastchessCommand,
) -> Result<EvaluationReport> {
    write_command_artifact(output_dir, &command)?;

    let run = fastchess::run(&command).with_context(|| {
        format!(
            "running Fastchess match; command saved to {}",
            output_dir.join("fastchess.command.txt").display()
        )
    })?;

    spec.fastchess_version = run.version.clone();

    persist_run_artifacts(output_dir, &run)?;
    validate_run(output_dir, &command, &run)?;

    let report = report_from_run(spec, &command, &run)?;

    write_report(output_dir, &report)?;

    Ok(report)
}

fn write_command_artifact(output_dir: &Path, command: &FastchessCommand) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    fs::write(
        output_dir.join("fastchess.command.txt"),
        format!("{}\n", command.command_line()?),
    )?;

    Ok(())
}

fn persist_run_artifacts(output_dir: &Path, run: &FastchessRun) -> Result<()> {
    fs::write(output_dir.join("fastchess.stdout.log"), &run.stdout)?;
    fs::write(output_dir.join("fastchess.stderr.log"), &run.stderr)?;

    if let Some(pgn) = &run.pgn {
        persist_clean_pgn(output_dir, pgn)?;
    }

    Ok(())
}

fn persist_clean_pgn(output_dir: &Path, pgn: &str) -> Result<()> {
    let clean_games: Vec<String> = split_pgn_games(pgn)
        .into_iter()
        .map(|game| clean_pgn(&game))
        .collect();
    let clean_pgn = clean_games.join("\n");

    fs::write(output_dir.join("games.pgn"), &clean_pgn)?;

    let games_dir = output_dir.join("games");
    fs::create_dir_all(&games_dir)?;
    for (index, game) in clean_games.into_iter().enumerate() {
        fs::write(games_dir.join(format!("game-{:04}.pgn", index + 1)), game)?;
    }

    Ok(())
}

fn validate_run(output_dir: &Path, command: &FastchessCommand, run: &FastchessRun) -> Result<()> {
    if run.status != 0 {
        bail!(
            "Fastchess exited with status {}; see {} and {}",
            run.status,
            output_dir.join("fastchess.stdout.log").display(),
            output_dir.join("fastchess.stderr.log").display()
        );
    }

    if let Some(pgn_output) = &command.pgn_output {
        if !output_dir.join("games.pgn").is_file() {
            bail!(
                "Fastchess completed without the requested PGN output {}; see {} and {}",
                pgn_output.display(),
                output_dir.join("fastchess.stdout.log").display(),
                output_dir.join("fastchess.stderr.log").display()
            );
        }
    }

    Ok(())
}

fn report_from_run(
    spec: EvaluationSpec,
    command: &FastchessCommand,
    run: &FastchessRun,
) -> Result<EvaluationReport> {
    let wdl = fastchess::parse_wdl(&run.stdout)
        .or_else(|| fastchess::parse_wdl(&run.stderr))
        .context("Fastchess completed but its candidate WDL summary could not be parsed")?;
    let score = Score {
        wins: wdl.wins as usize,
        draws: wdl.draws as usize,
        losses: wdl.losses as usize,
    };

    Ok(EvaluationReport::from_score(
        spec,
        score,
        report_artifacts(command),
    ))
}

fn report_artifacts(command: &FastchessCommand) -> Vec<String> {
    let mut artifacts = vec![
        "fastchess.command.txt".into(),
        "fastchess.stdout.log".into(),
        "fastchess.stderr.log".into(),
    ];
    if command.pgn_output.is_some() {
        artifacts.extend(["games.pgn".into(), "games/".into()]);
    }

    artifacts
}

/// Fastchess emits one PGN per game, separated by the Event tag. Preserve the
/// aggregate PGN and additionally make each game independently inspectable.
fn split_pgn_games(pgn: &str) -> Vec<String> {
    let mut games = Vec::new();
    let mut current = String::new();
    for line in pgn.lines() {
        if line.starts_with("[Event ") && !current.trim().is_empty() {
            games.push(std::mem::take(&mut current));
        }
        current.push_str(line);
        current.push('\n');
    }
    if !current.trim().is_empty() {
        games.push(current);
    }
    games
}

/// Keep only SAN movetext and the terminal result, dropping headers and
/// Fastchess' clock/statistics comments.
fn clean_pgn(pgn: &str) -> String {
    let mut white = None;
    let mut black = None;
    let mut result = None;
    for line in pgn.lines() {
        let value = |tag: &str| {
            line.strip_prefix(&format!("[{tag} \""))
                .and_then(|rest| rest.strip_suffix("\"]"))
                .map(str::to_owned)
        };
        white = white.or_else(|| value("White"));
        black = black.or_else(|| value("Black"));
        result = result.or_else(|| value("Result"));
    }
    let mut cleaned = String::new();
    let mut in_headers = true;
    let mut in_comment = false;
    for line in pgn.lines() {
        if in_headers {
            if line.trim().is_empty() {
                in_headers = false;
            }
            continue;
        }
        for ch in line.chars() {
            if ch == '{' {
                in_comment = true;
            }
            else if ch == '}' {
                in_comment = false;
            }
            else if !in_comment {
                cleaned.push(ch);
            }
        }
        cleaned.push(' ');
    }
    let moves = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut output = String::new();
    if let Some(white) = white {
        output.push_str(&format!("[White \"{white}\"]\n"));
    }
    if let Some(black) = black {
        output.push_str(&format!("[Black \"{black}\"]\n"));
    }
    output.push('\n');
    output.push_str(&moves);
    if let Some(result) = result {
        if !moves.ends_with(&result) {
            output.push(' ');
            output.push_str(&result);
        }
    }
    output.push('\n');
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fastchess::Engine;
    use crate::report::{EngineSpec, SearchSpec, SCHEMA_VERSION};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn failed_match_persists_fastchess_diagnostics() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("match-suite-failure-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake-fastchess.sh");
        fs::write(
            &script,
            "#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then exit 0; fi\necho captured-out\necho captured-err >&2\nexit 7\n",
        )
        .unwrap();
        std::process::Command::new("chmod")
            .args(["+x", script.to_str().unwrap()])
            .status()
            .unwrap();
        let output = dir.join("artifacts");
        let engine = |name: &str| EngineSpec {
            name: name.into(),
            command: name.into(),
            args: Vec::new(),
            checkpoint: None,
            options: Vec::new(),
        };
        let spec = EvaluationSpec {
            schema_version: SCHEMA_VERSION,
            id: "failure-test".into(),
            suite: "arena".into(),
            candidate: engine("candidate"),
            opponent: Some(engine("opponent")),
            search: SearchSpec {
                simulations: 1,
                temperature: 0.0,
                dirichlet_noise: false,
                device: "cpu".into(),
            },
            opening_set: None,
            seed: None,
            concurrency: 1,
            fastchess_version: None,
        };
        let command = FastchessCommand::new(
            &script,
            Engine::new("candidate", "candidate"),
            Engine::new("opponent", "opponent"),
        );

        let error = run_match(&output, spec, command).unwrap_err().to_string();
        assert!(error.contains("status 7"));
        assert_eq!(
            fs::read_to_string(output.join("fastchess.stdout.log")).unwrap(),
            "captured-out\n"
        );
        assert_eq!(
            fs::read_to_string(output.join("fastchess.stderr.log")).unwrap(),
            "captured-err\n"
        );
        assert!(output.join("fastchess.command.txt").is_file());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn pgn_cleanup_splits_games_and_removes_fastchess_comments() {
        let pgn = r#"[Event "one"]
[White "candidate"]
[Black "baseline"]
[Result "1-0"]

1. e4 {book, 0.01} e5 2. Nf3 1-0
[Event "two"]
[White "baseline"]
[Black "candidate"]
[Result "1/2-1/2"]

1. d4 {clk 0:01} d5 1/2-1/2
"#;
        let games = split_pgn_games(pgn);
        assert_eq!(games.len(), 2);
        let first = clean_pgn(&games[0]);
        assert!(first.contains("[White \"candidate\"]"));
        assert!(first.contains("1. e4 e5 2. Nf3 1-0"));
        assert!(!first.contains("book"));
        let second = clean_pgn(&games[1]);
        assert!(second.contains("1. d4 d5 1/2-1/2"));
    }

    #[test]
    fn successful_match_requires_the_requested_pgn_artifact() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("match-suite-missing-pgn-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("fake-fastchess.sh");
        fs::write(
            &script,
            "#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then exit 0; fi\necho 'Score of candidate vs baseline: 1 - 0 - 0 [1.000] 1'\n",
        )
        .unwrap();
        std::process::Command::new("chmod")
            .args(["+x", script.to_str().unwrap()])
            .status()
            .unwrap();
        let output = dir.join("artifacts");
        let engine = |name: &str| EngineSpec {
            name: name.into(),
            command: name.into(),
            args: Vec::new(),
            checkpoint: None,
            options: Vec::new(),
        };
        let spec = EvaluationSpec {
            schema_version: SCHEMA_VERSION,
            id: "missing-pgn".into(),
            suite: "arena".into(),
            candidate: engine("candidate"),
            opponent: Some(engine("baseline")),
            search: SearchSpec {
                simulations: 1,
                temperature: 0.0,
                dirichlet_noise: false,
                device: "cpu".into(),
            },
            opening_set: None,
            seed: None,
            concurrency: 1,
            fastchess_version: None,
        };
        let command = FastchessCommand::new(
            &script,
            Engine::new("candidate", "candidate"),
            Engine::new("baseline", "baseline"),
        )
        .pgn_output(output.join("games.pgn"));
        let error = run_match(&output, spec, command).unwrap_err().to_string();
        assert!(error.contains("without the requested PGN"));
        assert!(output.join("fastchess.stdout.log").is_file());
        fs::remove_dir_all(dir).unwrap();
    }
}
