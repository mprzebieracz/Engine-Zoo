use super::*;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn builds_reproducible_paired_opening_command() {
    let command = FastchessCommand::new(
        "fastchess",
        Engine::new("candidate", "candidate"),
        Engine::new("reference", "reference")
            .option("Threads", "2")
            .nodes(2_000_000),
    )
    .rounds(3)
    .concurrency(4)
    .openings(Openings::pgn("book.pgn").plies(8))
    .pgn_output("games.pgn");
    assert_eq!(
        command.args().unwrap(),
        vec![
            "-engine",
            "cmd=candidate",
            "name=candidate",
            "-engine",
            "cmd=reference",
            "name=reference",
            "option.Threads=2",
            "nodes=2000000",
            "-each",
            "tc=10+0.1",
            "-rounds",
            "3",
            "-repeat",
            "-concurrency",
            "4",
            "-openings",
            "file=book.pgn",
            "format=pgn",
            "order=sequential",
            "plies=8",
            "-pgnout",
            "file=games.pgn",
            "notation=san",
            "append=false",
        ]
    );
}

#[test]
fn parses_common_fastchess_summaries() {
    let text =
        "Score of candidate vs reference: 3 - 2 - 5 [0.550] 10\nWins: 3, Losses: 2, Draws: 5";
    assert_eq!(
        parse_wdl(text),
        Some(Wdl {
            wins: 3,
            draws: 5,
            losses: 2
        })
    );
    assert_eq!(parse_score(text), Some(0.55));
}

#[test]
fn parses_percentage_scores_and_rejects_unrelated_numbers() {
    assert_eq!(parse_score("Score: 62.5%"), Some(0.625));
    assert_eq!(parse_score("Games: 10\nElo: 25"), None);
    assert_eq!(
        parse_wdl("Wins: 0, Losses: 0, Draws: 0"),
        Some(Wdl::default())
    );
}

#[test]
fn validates_commands_and_quotes_diagnostic_command_lines() {
    let duplicate = FastchessCommand::new(
        "fast chess",
        Engine::new("engine one", "same"),
        Engine::new("engine two", "same"),
    );
    assert!(duplicate.args().unwrap_err().to_string().contains("names"));

    let zero_rounds =
        FastchessCommand::new("fastchess", Engine::new("a", "a"), Engine::new("b", "b")).rounds(0);
    assert!(zero_rounds
        .args()
        .unwrap_err()
        .to_string()
        .contains("rounds"));

    let zero_concurrency =
        FastchessCommand::new("fastchess", Engine::new("a", "a"), Engine::new("b", "b"))
            .concurrency(0);
    assert!(zero_concurrency
        .args()
        .unwrap_err()
        .to_string()
        .contains("concurrency"));

    let zero_nodes = FastchessCommand::new(
        "fastchess",
        Engine::new("a", "a").nodes(0),
        Engine::new("b", "b"),
    );
    assert!(zero_nodes.args().unwrap_err().to_string().contains("nodes"));

    let quoted = FastchessCommand::new(
        "fast chess",
        Engine::new("candidate's engine", "candidate"),
        Engine::new("reference", "reference"),
    )
    .command_line()
    .unwrap();
    assert!(quoted.starts_with("'fast chess'"));
    assert!(quoted.contains("'cmd=candidate'\\''s engine'"));
}

#[test]
fn runs_fake_binary_and_captures_pgn() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("fastchess-test-{unique}"));
    fs::create_dir_all(&dir).unwrap();
    let script = dir.join("fake-fastchess.sh");
    let pgn = dir.join("games.pgn");
    fs::write(&script, format!("#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then echo 'Fastchess 1.2.3'; exit 0; fi\necho 'Score of candidate vs reference: 1 - 0 - 1 [0.750] 2'\necho warning >&2\nprintf '[Result \\\"1/2-1/2\\\"]\\n' > '{}'\n", pgn.display())).unwrap();
    std::process::Command::new("chmod")
        .args(["+x", script.to_str().unwrap()])
        .status()
        .unwrap();
    let result = run(
        &FastchessCommand::new(&script, Engine::new("a", "a"), Engine::new("b", "b"))
            .pgn_output(&pgn),
    )
    .unwrap();
    assert_eq!(result.version.as_deref(), Some("Fastchess 1.2.3"));
    assert_eq!(result.pgn.as_deref(), Some("[Result \"1/2-1/2\"]\n"));
    assert_eq!(result.stderr.trim(), "warning");
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn captures_stdout_and_stderr_when_fastchess_fails() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("fastchess-failure-test-{unique}"));
    fs::create_dir_all(&dir).unwrap();
    let script = dir.join("fake-fastchess.sh");
    fs::write(
            &script,
            "#!/bin/sh\necho 'invalid opening settings'\necho 'usage: -openings file=FILE format=epd' >&2\nexit 1\n",
        )
        .unwrap();
    std::process::Command::new("chmod")
        .args(["+x", script.to_str().unwrap()])
        .status()
        .unwrap();

    let run = run(&FastchessCommand::new(
        &script,
        Engine::new("a", "a"),
        Engine::new("b", "b"),
    ))
    .unwrap();
    assert_eq!(run.status, 1);
    assert!(run.stdout.contains("invalid opening settings"));
    assert!(run.stderr.contains("usage: -openings file=FILE format=epd"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn removes_a_stale_pgn_before_starting_a_match() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("fastchess-stale-pgn-test-{unique}"));
    fs::create_dir_all(&dir).unwrap();
    let script = dir.join("fake-fastchess.sh");
    let pgn = dir.join("games.pgn");
    fs::write(&pgn, "stale game\n").unwrap();
    fs::write(
        &script,
        "#!/bin/sh\nif [ \"$1\" = \"-version\" ]; then exit 0; fi\nexit 3\n",
    )
    .unwrap();
    std::process::Command::new("chmod")
        .args(["+x", script.to_str().unwrap()])
        .status()
        .unwrap();

    let result = run(
        &FastchessCommand::new(&script, Engine::new("a", "a"), Engine::new("b", "b"))
            .pgn_output(&pgn),
    )
    .unwrap();
    assert_eq!(result.status, 3);
    assert_eq!(result.pgn, None);
    assert!(!pgn.exists());
    fs::remove_dir_all(dir).unwrap();
}
