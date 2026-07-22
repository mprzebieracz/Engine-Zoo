use super::*;

#[derive(Clone)]
pub(super) struct Bins {
    pub(super) puzzle: String,
    pub(super) puzzle_path: PathBuf,
    pub(super) stockfish: String,
    pub(super) stockfish_path: PathBuf,
    pub(super) arena: String,
    pub(super) arena_path: PathBuf,
    pub(super) fastchess: String,
    pub(super) fastchess_path: PathBuf,
    pub(super) uci: String,
    pub(super) uci_path: PathBuf,
    pub(super) stockfish_engine: String,
    pub(super) stockfish_engine_path: PathBuf,
    pub(super) openings: PathBuf,
    pub(super) puzzles: PathBuf,
}
impl Bins {
    pub(super) fn discover() -> Self {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let bin = |name: &str| {
            let p = root.join("target/release").join(name);
            if p.is_file() {
                p
            } else {
                root.join("target/debug").join(name)
            }
        };
        let env_or = |key: &str, fallback: PathBuf| {
            std::env::var_os(key).map(PathBuf::from).unwrap_or(fallback)
        };
        let fast = env_or(
            "FASTCHESS_BIN",
            root.join("crates/evaluations/bin/fastchess/fastchess"),
        );
        let sf = env_or(
            "STOCKFISH_BIN",
            root.join("crates/evaluations/bin/stockfish-18/stockfish-ubuntu-x86-64"),
        );
        Self {
            puzzle: bin("eval-puzzle").display().to_string(),
            puzzle_path: bin("eval-puzzle"),
            stockfish: bin("eval-stockfish").display().to_string(),
            stockfish_path: bin("eval-stockfish"),
            arena: bin("eval-arena").display().to_string(),
            arena_path: bin("eval-arena"),
            fastchess: fast.display().to_string(),
            fastchess_path: fast,
            uci: bin("engine-zoo-uci").display().to_string(),
            uci_path: bin("engine-zoo-uci"),
            stockfish_engine: sf.display().to_string(),
            stockfish_engine_path: sf,
            openings: env_or(
                "EVAL_OPENINGS",
                root.join("crates/evaluations/suites/chess/openings-balanced-v1.epd"),
            ),
            puzzles: env_or(
                "EVAL_PUZZLES",
                root.join("crates/evaluations/suites/chess/puzzles-smoke.jsonl"),
            ),
        }
    }
}

pub(super) fn read_result(suite: &Suite, output: &Path) -> Result<Value> {
    match suite {
        Suite::Puzzle => {
            let summary = BufReader::new(File::open(output.join("puzzles.jsonl"))?)
                .lines()
                .map_while(Result::ok)
                .filter(|line| !line.trim().is_empty())
                .last()
                .context("puzzle output contained no summary")?;
            Ok(serde_json::from_str(&summary)?)
        }
        Suite::Stockfish | Suite::Arena => Ok(serde_json::from_reader(File::open(
            output.join("report.json"),
        )?)?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn puzzle_result_uses_the_last_nonempty_jsonl_record() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("evaluation-result-{unique}"));
        fs::create_dir_all(&dir).unwrap();
        fs::write(
            dir.join("puzzles.jsonl"),
            "{\"record_type\":\"puzzle\"}\n\n{\"record_type\":\"summary\",\"total\":1}\n",
        )
        .unwrap();
        let value = read_result(&Suite::Puzzle, &dir).unwrap();
        assert_eq!(value["record_type"], "summary");
        assert_eq!(value["total"], 1);
        fs::remove_dir_all(dir).unwrap();
    }
}
