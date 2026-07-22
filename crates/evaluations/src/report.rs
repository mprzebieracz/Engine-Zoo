use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchSpec {
    pub simulations: usize,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub dirichlet_noise: bool,
    #[serde(default = "default_device")]
    pub device: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineSpec {
    pub name: String,
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub checkpoint: Option<String>,
    #[serde(default)]
    pub options: Vec<(String, String)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationSpec {
    pub schema_version: u32,
    pub id: String,
    pub suite: String,
    pub candidate: EngineSpec,
    #[serde(default)]
    pub opponent: Option<EngineSpec>,
    pub search: SearchSpec,
    #[serde(default)]
    pub opening_set: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub concurrency: usize,
    #[serde(default)]
    pub fastchess_version: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Score {
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
}

impl Score {
    pub fn games(self) -> usize {
        self.wins + self.draws + self.losses
    }

    pub fn points(self) -> f64 {
        self.wins as f64 + self.draws as f64 * 0.5
    }

    pub fn fraction(self) -> f64 {
        let games = self.games();
        if games == 0 {
            0.0
        } else {
            self.points() / games as f64
        }
    }

    /// A finite, project-local Elo estimate using a Jeffreys-style half point
    /// smoothing. It is meaningful only against the named opponent in a
    /// report with the same specification.
    pub fn smoothed_elo_delta(self) -> f64 {
        let games = self.games() as f64;
        let score = (self.points() + 0.5) / (games + 1.0);
        400.0 * (score / (1.0 - score)).log10()
    }

    /// Wilson interval for the score fraction. Draws are treated as half a
    /// point, which is intentionally a simple monitoring estimate rather than
    /// a replacement for Fastchess pentanomial/SPRT statistics.
    pub fn wilson_95(self) -> (f64, f64) {
        let n = self.games() as f64;
        if n == 0.0 {
            return (0.0, 0.0);
        }
        let z = 1.959_963_984_540_054;
        let p = self.fraction();
        let denom = 1.0 + z * z / n;
        let center = (p + z * z / (2.0 * n)) / denom;
        let radius = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt()) / denom;
        ((center - radius).max(0.0), (center + radius).min(1.0))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub schema_version: u32,
    pub id: String,
    pub suite: String,
    pub spec: EvaluationSpec,
    pub score: Score,
    pub score_fraction: f64,
    pub score_wilson_95: (f64, f64),
    pub smoothed_elo_delta: f64,
    pub completed_at_unix_s: u64,
    #[serde(default)]
    pub artifacts: Vec<String>,
}

impl EvaluationReport {
    pub fn from_score(spec: EvaluationSpec, score: Score, artifacts: Vec<String>) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            id: spec.id.clone(),
            suite: spec.suite.clone(),
            score_fraction: score.fraction(),
            score_wilson_95: score.wilson_95(),
            smoothed_elo_delta: score.smoothed_elo_delta(),
            completed_at_unix_s: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock predates Unix epoch")
                .as_secs(),
            spec,
            score,
            artifacts,
        }
    }
}

pub fn write_report(output_dir: &Path, report: &EvaluationReport) -> Result<()> {
    fs::create_dir_all(output_dir)?;
    fs::write(
        output_dir.join("spec.json"),
        serde_json::to_string_pretty(&report.spec)?,
    )?;
    fs::write(
        output_dir.join("report.json"),
        serde_json::to_string_pretty(report)?,
    )?;
    let index = output_dir
        .parent()
        .unwrap_or(output_dir)
        .join("index.jsonl");
    let mut file = OpenOptions::new().create(true).append(true).open(index)?;
    writeln!(file, "{}", serde_json::to_string(report)?)?;
    Ok(())
}

fn default_device() -> String {
    "cpu".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn score_statistics_handle_empty_and_perfect_matches() {
        assert_eq!(Score::default().wilson_95(), (0.0, 0.0));
        assert!(Score {
            wins: 4,
            draws: 0,
            losses: 0
        }
        .smoothed_elo_delta()
        .is_finite());
    }

    #[test]
    fn balanced_match_has_near_zero_delta() {
        assert!(
            Score {
                wins: 2,
                draws: 0,
                losses: 2
            }
            .smoothed_elo_delta()
            .abs()
                < 0.001
        );
    }

    fn spec(id: &str) -> EvaluationSpec {
        EvaluationSpec {
            schema_version: SCHEMA_VERSION,
            id: id.into(),
            suite: "arena".into(),
            candidate: EngineSpec {
                name: "candidate".into(),
                command: "uci".into(),
                args: vec!["--flag".into()],
                checkpoint: Some("candidate.safetensors".into()),
                options: vec![("Threads".into(), "1".into())],
            },
            opponent: None,
            search: SearchSpec {
                simulations: 100,
                temperature: 0.0,
                dirichlet_noise: false,
                device: "cpu".into(),
            },
            opening_set: None,
            seed: Some(7),
            concurrency: 1,
            fastchess_version: None,
        }
    }

    #[test]
    fn report_round_trips_and_appends_to_the_parent_index() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("evaluation-report-{unique}"));
        let output = root.join("first");
        let report = EvaluationReport::from_score(
            spec("first"),
            Score {
                wins: 3,
                draws: 2,
                losses: 1,
            },
            vec!["games.pgn".into()],
        );
        write_report(&output, &report).unwrap();

        let stored: EvaluationReport =
            serde_json::from_slice(&fs::read(output.join("report.json")).unwrap()).unwrap();
        assert_eq!(stored.id, "first");
        assert_eq!(stored.score, report.score);
        assert_eq!(stored.score_fraction, 4.0 / 6.0);
        let index = fs::read_to_string(root.join("index.jsonl")).unwrap();
        assert_eq!(index.lines().count(), 1);
        assert_eq!(
            serde_json::from_str::<EvaluationReport>(&index).unwrap().id,
            "first"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn wilson_interval_contains_the_observed_fraction() {
        for score in [
            Score {
                wins: 10,
                draws: 0,
                losses: 0,
            },
            Score {
                wins: 2,
                draws: 6,
                losses: 2,
            },
            Score {
                wins: 0,
                draws: 0,
                losses: 10,
            },
        ] {
            let (low, high) = score.wilson_95();
            assert!(low <= score.fraction() + f64::EPSILON);
            assert!(score.fraction() <= high + f64::EPSILON);
            assert!((0.0..=1.0).contains(&low));
            assert!((0.0..=1.0).contains(&high));
        }
    }
}
