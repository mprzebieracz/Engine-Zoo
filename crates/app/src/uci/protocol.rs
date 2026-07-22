pub const STARTPOS: &str = "startpos";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UciCommand {
    Uci,
    IsReady,
    UciNewGame,
    Position {
        fen: Option<String>,
        moves: Vec<String>,
    },
    Go {
        movetime_ms: Option<u64>,
        nodes: Option<usize>,
    },
    Stop,
    Quit,
    SetOption {
        name: String,
        value: String,
    },
    Unknown,
}

pub fn parse_command(line: &str) -> UciCommand {
    let mut words = line.split_whitespace();
    let Some(command) = words.next() else {
        return UciCommand::Unknown;
    };
    match command.to_ascii_lowercase().as_str() {
        "uci" => UciCommand::Uci,
        "isready" => UciCommand::IsReady,
        "ucinewgame" => UciCommand::UciNewGame,
        "stop" => UciCommand::Stop,
        "quit" => UciCommand::Quit,
        "setoption" => parse_setoption(words.collect()),
        "position" => parse_position(words.collect()),
        "go" => parse_go(words.collect()),
        _ => UciCommand::Unknown,
    }
}

fn parse_setoption(words: Vec<&str>) -> UciCommand {
    let Some(name_at) = words
        .iter()
        .position(|word| word.eq_ignore_ascii_case("name"))
    else {
        return UciCommand::Unknown;
    };
    let value_at = words
        .iter()
        .position(|word| word.eq_ignore_ascii_case("value"));
    let end = value_at.unwrap_or(words.len());
    if name_at + 1 >= end {
        return UciCommand::Unknown;
    }
    let name = words[name_at + 1..end].join(" ");
    let value = value_at.map_or_else(String::new, |at| words[at + 1..].join(" "));
    UciCommand::SetOption { name, value }
}

fn parse_position(words: Vec<&str>) -> UciCommand {
    let Some(first) = words.first().copied() else {
        return UciCommand::Unknown;
    };
    let (fen, moves_at) = if first.eq_ignore_ascii_case(STARTPOS) {
        (None, 1)
    } else if first.eq_ignore_ascii_case("fen") {
        if words.len() < 7 {
            return UciCommand::Unknown;
        }
        (Some(words[1..7].join(" ")), 7)
    } else {
        return UciCommand::Unknown;
    };
    let moves = if words.get(moves_at) == Some(&"moves") {
        words[moves_at + 1..]
            .iter()
            .map(|s| (*s).to_owned())
            .collect()
    } else if moves_at == words.len() {
        Vec::new()
    } else {
        return UciCommand::Unknown;
    };
    UciCommand::Position { fen, moves }
}

fn parse_go(words: Vec<&str>) -> UciCommand {
    let mut movetime_ms = None;
    let mut nodes = None;
    for pair in words.windows(2) {
        match pair[0].to_ascii_lowercase().as_str() {
            "movetime" => movetime_ms = pair[1].parse().ok(),
            "nodes" => nodes = pair[1].parse().ok(),
            _ => {}
        }
    }
    UciCommand::Go { movetime_ms, nodes }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_positions_and_go_limits() {
        assert_eq!(
            parse_command("position startpos moves e2e4 e7e5"),
            UciCommand::Position {
                fen: None,
                moves: vec!["e2e4".into(), "e7e5".into()]
            }
        );
        assert_eq!(
            parse_command("position fen 8/8/8/8/8/8/4K3/4k3 w - - 0 1 moves e2e3"),
            UciCommand::Position {
                fen: Some("8/8/8/8/8/8/4K3/4k3 w - - 0 1".into()),
                moves: vec!["e2e3".into()]
            }
        );
        assert_eq!(
            parse_command("go movetime 50 nodes 12"),
            UciCommand::Go {
                movetime_ms: Some(50),
                nodes: Some(12)
            }
        );
    }

    #[test]
    fn parses_commands_case_insensitively_and_preserves_option_text() {
        assert_eq!(parse_command(" UCI "), UciCommand::Uci);
        assert_eq!(parse_command("ISREADY"), UciCommand::IsReady);
        assert_eq!(
            parse_command("setoption name Run Directory value data/my run"),
            UciCommand::SetOption {
                name: "Run Directory".into(),
                value: "data/my run".into(),
            }
        );
        assert_eq!(
            parse_command("setoption name Clear Hash"),
            UciCommand::SetOption {
                name: "Clear Hash".into(),
                value: String::new(),
            }
        );
    }

    #[test]
    fn rejects_malformed_positions_and_options() {
        for command in [
            "",
            "position",
            "position startpos trailing",
            "position fen 8/8/8/8/8/8/8/K6k w - - 0",
            "position unknown",
            "setoption value 1",
            "setoption name value 1",
        ] {
            assert_eq!(parse_command(command), UciCommand::Unknown, "{command}");
        }
    }

    #[test]
    fn go_ignores_unknown_limits_without_losing_known_ones() {
        assert_eq!(
            parse_command("go wtime 1000 nodes 42 ponder true movetime 25"),
            UciCommand::Go {
                movetime_ms: Some(25),
                nodes: Some(42),
            }
        );
        assert_eq!(
            parse_command("go nodes invalid"),
            UciCommand::Go {
                movetime_ms: None,
                nodes: None,
            }
        );
    }

    #[test]
    fn go_finds_limits_after_variable_length_searchmoves() {
        assert_eq!(
            parse_command("go searchmoves e2e4 d2d4 infinite nodes 42 movetime 25"),
            UciCommand::Go {
                movetime_ms: Some(25),
                nodes: Some(42),
            }
        );
    }
}
