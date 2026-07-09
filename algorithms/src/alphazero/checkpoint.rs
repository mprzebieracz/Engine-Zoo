use super::network::NetConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub game: String,
    pub net: NetConfig,
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
        } else {
            fs::create_dir_all(run.checkpoints_dir())?;
            let cfg = make_config();
            fs::write(&cfg_path, serde_json::to_string_pretty(&cfg)?)?;
            cfg
        };
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
