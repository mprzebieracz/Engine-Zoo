use super::network::{ChessAzV2Config, NetConfig, NetworkConfig};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub game: String,
    pub net: NetConfig,
    /// Missing in pre-v2 config files, and therefore defaults to legacy.
    #[serde(default)]
    pub architecture: RunArchitecture,
}

/// Explicit checkpoint/data format selected by a run. It is serialized as a
/// tagged enum, so a v2 run cannot be mistaken for legacy based on dimensions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "kebab-case")]
pub enum RunArchitecture {
    #[default]
    Legacy,
    ChessAzV2(ChessAzV2Config),
}

impl RunConfig {
    pub fn network_config(&self) -> NetworkConfig {
        match self.architecture {
            RunArchitecture::Legacy => NetworkConfig::Legacy(self.net.clone()),
            RunArchitecture::ChessAzV2(cfg) => NetworkConfig::ChessAzV2(cfg),
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self.architecture {
            RunArchitecture::Legacy => Ok(()),
            RunArchitecture::ChessAzV2(cfg) => {
                anyhow::ensure!(
                    self.game == "chess",
                    "chess-az-v2 is only valid for chess runs"
                );
                cfg.validate()
            }
        }
    }
}

/// A training run directory:
///
/// ```text
/// <root>/config.json                     game + network architecture
/// <root>/checkpoints/ckpt_0000.safetensors ...
/// <root>/archive/ckpt_iter_000001_1234567890.safetensors
/// <root>/best.safetensors                network used for self-play
/// <root>/candidate.safetensors           gated mode's challenger
/// <root>/metrics.jsonl                   one JSON record per iteration
/// ```
pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    /// Opens an existing run (validating its config exists) or creates a new
    /// one with `make_config`.
    pub fn open_or_create(
        root: &Path,
        make_config: impl FnOnce() -> RunConfig,
    ) -> Result<(RunDir, RunConfig)> {
        let run = RunDir {
            root: root.to_path_buf(),
        };
        let cfg_path = run.root.join("config.json");
        let cfg = if cfg_path.exists() {
            serde_json::from_str(&fs::read_to_string(&cfg_path)?)
                .with_context(|| format!("parsing {}", cfg_path.display()))?
        }
        else {
            fs::create_dir_all(run.checkpoints_dir())?;
            let cfg = make_config();
            fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;
            cfg
        };
        cfg.validate()?;
        fs::create_dir_all(run.checkpoints_dir())?;
        Ok((run, cfg))
    }

    fn checkpoints_dir(&self) -> PathBuf {
        self.root.join("checkpoints")
    }

    fn archive_dir(&self) -> PathBuf {
        self.root.join("archive")
    }

    pub fn checkpoint_path(&self, index: u32) -> PathBuf {
        self.checkpoints_dir()
            .join(format!("ckpt_{index:04}.safetensors"))
    }

    pub fn archive_checkpoint_path(&self, iteration: usize, unix_secs: u64) -> Result<PathBuf> {
        let dir = self.archive_dir();
        fs::create_dir_all(&dir)?;
        Ok(dir.join(format!("ckpt_iter_{iteration:06}_{unix_secs}.safetensors")))
    }

    pub fn best_path(&self) -> PathBuf {
        self.root.join("best.safetensors")
    }

    pub fn candidate_path(&self) -> PathBuf {
        self.root.join("candidate.safetensors")
    }

    /// Highest existing checkpoint index, if any.
    pub fn latest_checkpoint(&self) -> Option<(u32, PathBuf)> {
        let entries = fs::read_dir(self.checkpoints_dir()).ok()?;
        entries
            .flatten()
            .filter_map(|e| {
                let name = e.file_name().into_string().ok()?;
                let idx: u32 = name
                    .strip_prefix("ckpt_")?
                    .strip_suffix(".safetensors")?
                    .parse()
                    .ok()?;
                Some((idx, e.path()))
            })
            .max_by_key(|(idx, _)| *idx)
    }

    /// Weights to resume training from. `best.safetensors` is updated after
    /// every continuous-training iteration, while numbered checkpoints may be
    /// intentionally sparse, so resuming from the latter can silently discard
    /// successful updates.
    pub fn training_checkpoint(&self) -> Option<PathBuf> {
        let best = self.best_path();
        if best.is_file() {
            Some(best)
        }
        else {
            self.latest_checkpoint().map(|(_, path)| path)
        }
    }

    /// Appends one JSON record (with a timestamp) to `metrics.jsonl`.
    pub fn log_metrics(&self, mut record: serde_json::Value) -> Result<()> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        record["time"] = now.into();
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("metrics.jsonl"))?;
        writeln!(f, "{record}")?;
        println!("metrics: {record}");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alphazero::representation::Connect4AzRepresentation;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn test_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "engine-zoo-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn config() -> RunConfig {
        RunConfig {
            game: "connect4".into(),
            net: NetConfig::for_representation::<games::Connect4, Connect4AzRepresentation>(1, 8),
            architecture: RunArchitecture::Legacy,
        }
    }

    #[test]
    fn training_resume_prefers_best_over_sparse_numbered_checkpoint() {
        let root = test_root("training-resume");
        let (run, _) = RunDir::open_or_create(&root, config).unwrap();
        let numbered = run.checkpoint_path(0);
        fs::write(&numbered, "initial").unwrap();
        fs::write(run.best_path(), "newer").unwrap();

        assert_eq!(run.training_checkpoint(), Some(run.best_path()));
        assert_eq!(
            fs::read(run.training_checkpoint().unwrap()).unwrap(),
            b"newer"
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn training_resume_falls_back_to_numbered_checkpoint_for_old_runs() {
        let root = test_root("training-resume-fallback");
        let (run, _) = RunDir::open_or_create(&root, config).unwrap();
        let checkpoint = run.checkpoint_path(7);
        fs::write(&checkpoint, "checkpoint").unwrap();

        assert_eq!(run.training_checkpoint(), Some(checkpoint));

        fs::remove_dir_all(root).unwrap();
    }
}
