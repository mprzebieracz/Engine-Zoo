use super::migration::migrate_old_config;
use super::state::RunStateV1;
use super::{
    migration::{migrate_version_two, VersionTwoExperiment},
    ExperimentConfig, RunState, STATE_FORMAT_VERSION,
};
use anyhow::{Context, Result};
use std::ffi::OsString;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    /// Opens an initialized run without inventing configuration or state.
    pub fn open(root: &Path) -> Result<(Self, ExperimentConfig, RunState)> {
        let run = Self {
            root: root.to_path_buf(),
        };
        anyhow::ensure!(
            run.experiment_path().is_file(),
            "missing immutable experiment file: {}",
            run.experiment_path().display()
        );
        anyhow::ensure!(
            run.state_path().is_file(),
            "missing run state: {}",
            run.state_path().display()
        );
        Ok((
            run,
            read_experiment(&root.join("experiment.toml"))?,
            read_state(&root.join("state.json"))?,
        ))
    }

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
        }
        else if run.legacy_experiment_path().is_file() {
            let migrated = read_legacy_experiment(&run.legacy_experiment_path())?;
            write_toml_atomic(&run.experiment_path(), &migrated)?;
            migrated
        }
        else if run.old_config_path().is_file() {
            let migrated = migrate_old_config(&fs::read_to_string(run.old_config_path())?)?;
            migrated.validate()?;
            write_toml_atomic(&run.experiment_path(), &migrated)?;
            migrated
        }
        else {
            let config = make_config();
            config.validate()?;
            write_toml_atomic(&run.experiment_path(), &config)?;
            config
        };
        config.validate()?;
        let state = if run.state_path().is_file() {
            read_state(&run.state_path())?
        }
        else {
            let state = RunState::default();
            run.write_state(&state)?;
            state
        };
        Ok((run, config, state))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Creates a run from a validated TOML experiment. Existing experiments
    /// are never overwritten: algorithm and model settings are immutable.
    pub fn initialize(root: &Path, config: ExperimentConfig) -> Result<(Self, RunState)> {
        let run = Self {
            root: root.to_path_buf(),
        };
        if root.exists() {
            anyhow::ensure!(
                fs::read_dir(root)?.next().is_none(),
                "run directory is not empty: {}",
                root.display()
            );
        }
        anyhow::ensure!(
            !run.experiment_path().exists()
                && !run.legacy_experiment_path().exists()
                && !run.old_config_path().exists(),
            "run directory already contains an experiment: {}",
            root.display()
        );
        config.validate()?;
        fs::create_dir_all(run.checkpoints_dir())?;
        write_toml_atomic(&run.experiment_path(), &config)?;
        let state = RunState::default();
        run.write_state(&state)?;
        Ok((run, state))
    }

    pub fn experiment_path(&self) -> PathBuf {
        self.root.join("experiment.toml")
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
        let temporary = temporary_model_path(&latest)?;
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

    fn legacy_experiment_path(&self) -> PathBuf {
        self.root.join("experiment.json")
    }
}

fn read_experiment(path: &Path) -> Result<ExperimentConfig> {
    let config: ExperimentConfig = toml::from_str(&fs::read_to_string(path)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

fn read_legacy_experiment(path: &Path) -> Result<ExperimentConfig> {
    let contents = fs::read_to_string(path)?;
    let version = serde_json::from_str::<serde_json::Value>(&contents)?
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .context("experiment format_version must be an unsigned integer")?;
    let config = match version {
        1 | 3 => serde_json::from_str::<ExperimentConfig>(&contents)
            .with_context(|| format!("parsing {}", path.display()))?,
        2 => migrate_version_two(
            serde_json::from_str::<VersionTwoExperiment>(&contents)
                .with_context(|| format!("parsing {}", path.display()))?,
        )?,
        version => anyhow::bail!("unsupported experiment format version {version}"),
    };
    let config = if config.format_version == 1 {
        config.upgrade_from_v1()?
    }
    else {
        config
    };
    config.validate()?;
    Ok(config)
}

fn read_state(path: &Path) -> Result<RunState> {
    let json = fs::read_to_string(path)?;
    let format_version = serde_json::from_str::<serde_json::Value>(&json)
        .with_context(|| format!("parsing {}", path.display()))?
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .context("state format_version must be an unsigned integer")?;
    match format_version {
        version if version == u64::from(STATE_FORMAT_VERSION) => {
            serde_json::from_str(&json).with_context(|| format!("parsing {}", path.display()))
        }
        1 => {
            let state: RunStateV1 = serde_json::from_str(&json)
                .with_context(|| format!("parsing {}", path.display()))?;
            anyhow::ensure!(state.format_version == 1, "invalid version-1 state");
            Ok(RunState::from_v1(state))
        }
        version => anyhow::bail!("unsupported state format version {version}"),
    }
}

fn temporary_model_path(final_path: &Path) -> Result<PathBuf> {
    let stem = final_path
        .file_stem()
        .context("checkpoint path must have a file stem")?;
    let extension = final_path
        .extension()
        .context("checkpoint path must have a file extension")?;
    let mut name = OsString::from(stem);
    name.push(".tmp.");
    name.push(extension);
    Ok(final_path.with_file_name(name))
}

fn write_json_atomic<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn write_toml_atomic<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, toml::to_string_pretty(value)?)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn relative_to_root(root: &Path, path: &Path) -> Result<String> {
    Ok(path.strip_prefix(root)?.to_string_lossy().into_owned())
}
