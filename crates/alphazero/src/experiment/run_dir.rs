use super::migration::migrate_old_config;
use super::{ExperimentConfig, RunState, STATE_FORMAT_VERSION};
use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    pub fn open_or_create(
        root: &Path,
        make_config: impl FnOnce() -> ExperimentConfig,
    ) -> Result<(Self, ExperimentConfig, RunState)> {
        let run = Self {
            root: root.to_path_buf(),
        };
        fs::create_dir_all(run.checkpoints_dir())?;
        let config = if run.experiment_path().is_file() {
            read_experiment(&run.experiment_path())?
        } else if run.old_config_path().is_file() {
            let migrated = migrate_old_config(&fs::read_to_string(run.old_config_path())?)?;
            migrated.validate()?;
            write_json_atomic(&run.experiment_path(), &migrated)?;
            migrated
        } else {
            let config = make_config();
            config.validate()?;
            write_json_atomic(&run.experiment_path(), &config)?;
            config
        };
        config.validate()?;
        let state = if run.state_path().is_file() {
            read_state(&run.state_path())?
        } else {
            let state = RunState::default();
            run.write_state(&state)?;
            state
        };
        Ok((run, config, state))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }
    pub fn experiment_path(&self) -> PathBuf {
        self.root.join("experiment.json")
    }
    pub fn state_path(&self) -> PathBuf {
        self.root.join("state.json")
    }
    pub fn latest_path(&self) -> PathBuf {
        self.checkpoints_dir().join("latest.safetensors")
    }
    pub fn candidate_path(&self) -> PathBuf {
        self.checkpoints_dir().join("candidate.safetensors")
    }
    pub fn best_path(&self) -> PathBuf {
        self.checkpoints_dir().join("best.safetensors")
    }
    pub fn checkpoints_dir(&self) -> PathBuf {
        self.root.join("checkpoints")
    }
    pub fn archived_checkpoint_path(&self, generation: u64) -> PathBuf {
        self.checkpoints_dir()
            .join(format!("generation-{generation:06}.safetensors"))
    }

    pub fn latest_checkpoint(&self, state: &RunState) -> Option<PathBuf> {
        state
            .latest_checkpoint
            .as_ref()
            .map(|path| self.root.join(path))
            .filter(|path| path.is_file())
    }

    /// Writes the model to a sibling temporary file, renames it into place,
    /// then records the new latest checkpoint in `state.json`.
    pub fn write_latest(
        &self,
        state: &mut RunState,
        write_model: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<()> {
        let latest = self.latest_path();
        let temporary = latest.with_extension("safetensors.tmp");
        let _ = fs::remove_file(&temporary);
        write_model(&temporary).with_context(|| format!("writing {}", temporary.display()))?;
        fs::rename(&temporary, &latest)?;
        state.latest_checkpoint = Some(relative_to_root(&self.root, &latest)?);
        self.write_state(state)
    }

    pub fn write_state(&self, state: &RunState) -> Result<()> {
        anyhow::ensure!(
            state.format_version == STATE_FORMAT_VERSION,
            "unsupported state format version {}",
            state.format_version
        );
        write_json_atomic(&self.state_path(), state)
    }

    pub fn log_metrics(&self, mut record: serde_json::Value) -> Result<()> {
        record["time"] = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64()
            .into();
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("metrics.jsonl"))?;
        writeln!(file, "{record}")?;
        Ok(())
    }

    fn old_config_path(&self) -> PathBuf {
        self.root.join("config.json")
    }
}

fn read_experiment(path: &Path) -> Result<ExperimentConfig> {
    let config: ExperimentConfig = serde_json::from_str(&fs::read_to_string(path)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

fn read_state(path: &Path) -> Result<RunState> {
    let state: RunState = serde_json::from_str(&fs::read_to_string(path)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    anyhow::ensure!(
        state.format_version == STATE_FORMAT_VERSION,
        "unsupported state format version {}",
        state.format_version
    );
    Ok(state)
}

fn write_json_atomic<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn relative_to_root(root: &Path, path: &Path) -> Result<String> {
    Ok(path.strip_prefix(root)?.to_string_lossy().into_owned())
}
