use super::migration::migrate_old_config;
use super::state::{CheckpointIdentity, RunStateV1, RunStateV2};
use super::{
    migration::{migrate_version_two, VersionTwoExperiment},
    ExperimentConfig, RunState, STATE_FORMAT_VERSION,
};
use anyhow::{Context, Result};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct RunDir {
    root: PathBuf,
    writer_lock: Option<File>,
}

impl RunDir {
    /// Opens an initialized run without inventing configuration or state.
    pub fn open(root: &Path) -> Result<(Self, ExperimentConfig, RunState)> {
        let run = Self {
            root: root.to_path_buf(),
            writer_lock: None,
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
            read_state(&root.join("state.json"), root)?,
        ))
    }

    pub fn open_writer(root: &Path) -> Result<(Self, ExperimentConfig, RunState)> {
        let (mut run, _, _) = Self::open(root)?;
        run.acquire_writer_lock()?;
        let config = read_experiment(&run.experiment_path())?;
        let mut state = read_state(&run.state_path(), root)?;
        run.make_checkpoint_generation_addressed(&mut state)?;

        Ok((run, config, state))
    }

    pub fn open_or_create(
        root: &Path,
        make_config: impl FnOnce() -> ExperimentConfig,
    ) -> Result<(Self, ExperimentConfig, RunState)> {
        let mut run = Self {
            root: root.to_path_buf(),
            writer_lock: None,
        };
        fs::create_dir_all(run.checkpoints_dir())?;
        run.acquire_writer_lock()?;
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
        let mut state = if run.state_path().is_file() {
            read_state(&run.state_path(), &run.root)?
        }
        else {
            let state = RunState::default();
            run.write_state(&state)?;
            state
        };
        run.make_checkpoint_generation_addressed(&mut state)?;

        Ok((run, config, state))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Creates a run from a validated TOML experiment. Existing experiments
    /// are never overwritten: algorithm and model settings are immutable.
    pub fn initialize(root: &Path, config: ExperimentConfig) -> Result<(Self, RunState)> {
        let mut run = Self {
            root: root.to_path_buf(),
            writer_lock: None,
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
        run.acquire_writer_lock()?;
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
            .checkpoint_identity
            .as_ref()
            .map(|identity| self.root.join(&identity.relative_path))
            .filter(|path| path.is_file())
    }

    /// Writes the model to a sibling temporary file, renames it into place,
    /// then records the new latest checkpoint in `state.json`.
    pub fn write_latest(
        &self,
        state: &mut RunState,
        write_model: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<()> {
        self.ensure_writer()?;

        let checkpoint = self.archived_checkpoint_path(state.model_generation);
        write_model_immutably(&checkpoint, write_model)?;

        state.checkpoint_identity = Some(CheckpointIdentity {
            generation: state.model_generation,
            relative_path: relative_to_root(&self.root, &checkpoint)?,
            sha256: crate::artifact::sha256_file(&checkpoint)?,
        });
        self.write_state(state)?;
        copy_atomically(&checkpoint, &self.latest_path())
    }

    /// Writes an immutable periodic snapshot without changing run state.
    pub fn write_archived(
        &self,
        generation: u64,
        write_model: impl FnOnce(&Path) -> Result<()>,
    ) -> Result<()> {
        self.ensure_writer()?;

        write_model_immutably(&self.archived_checkpoint_path(generation), write_model)
    }

    pub fn write_state(&self, state: &RunState) -> Result<()> {
        self.ensure_writer()?;

        anyhow::ensure!(
            state.format_version == STATE_FORMAT_VERSION,
            "unsupported state format version {}",
            state.format_version
        );
        write_json_atomic(&self.state_path(), state)
    }

    pub fn log_metrics(&self, mut record: serde_json::Value) -> Result<()> {
        self.ensure_writer()?;

        record["time"] = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64()
            .into();
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("metrics.jsonl"))?;
        writeln!(file, "{record}")?;
        file.flush()?;
        Ok(())
    }

    fn acquire_writer_lock(&mut self) -> Result<()> {
        let path = self.root.join(".writer.lock");
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(path)?;

        lock_exclusive(&file)?;
        self.writer_lock = Some(file);

        Ok(())
    }

    fn ensure_writer(&self) -> Result<()> {
        anyhow::ensure!(
            self.writer_lock.is_some(),
            "run directory is open read-only; open a writer before modifying it"
        );

        Ok(())
    }

    fn make_checkpoint_generation_addressed(&self, state: &mut RunState) -> Result<()> {
        let Some(identity) = &state.checkpoint_identity
        else {
            return Ok(());
        };
        let generation_path = self.archived_checkpoint_path(identity.generation);
        if self.root.join(&identity.relative_path) == generation_path {
            return Ok(());
        }

        let source = self.root.join(&identity.relative_path);
        write_model_immutably(&generation_path, |temporary| {
            fs::copy(&source, temporary)?;
            Ok(())
        })?;
        state.checkpoint_identity = Some(CheckpointIdentity {
            generation: identity.generation,
            relative_path: relative_to_root(&self.root, &generation_path)?,
            sha256: crate::artifact::sha256_file(&generation_path)?,
        });
        self.write_state(state)
    }

    fn old_config_path(&self) -> PathBuf {
        self.root.join("config.json")
    }

    fn legacy_experiment_path(&self) -> PathBuf {
        self.root.join("experiment.json")
    }
}

fn read_experiment(path: &Path) -> Result<ExperimentConfig> {
    ExperimentConfig::read_toml(path).with_context(|| format!("parsing {}", path.display()))
}

fn read_legacy_experiment(path: &Path) -> Result<ExperimentConfig> {
    let contents = fs::read_to_string(path)?;
    let version = serde_json::from_str::<serde_json::Value>(&contents)?
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .context("experiment format_version must be an unsigned integer")?;
    let config = match version {
        1 | 3 | 4 => serde_json::from_str::<ExperimentConfig>(&contents)
            .with_context(|| format!("parsing {}", path.display()))?,
        2 => migrate_version_two(
            serde_json::from_str::<VersionTwoExperiment>(&contents)
                .with_context(|| format!("parsing {}", path.display()))?,
        )?,
        version => anyhow::bail!("unsupported experiment format version {version}"),
    };
    let config = if matches!(config.format_version, 1 | 3) {
        config.upgrade_to_current()?
    }
    else {
        config
    };
    config.validate()?;
    Ok(config)
}

fn read_state(path: &Path, root: &Path) -> Result<RunState> {
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
            let (state, checkpoint) = RunState::from_v1(state);

            add_legacy_checkpoint_identity(state, checkpoint, root)
        }
        2 => {
            let state: RunStateV2 = serde_json::from_str(&json)
                .with_context(|| format!("parsing {}", path.display()))?;
            anyhow::ensure!(state.format_version == 2, "invalid version-2 state");
            let (state, checkpoint) = RunState::from_v2(state);

            add_legacy_checkpoint_identity(state, checkpoint, root)
        }
        version => anyhow::bail!("unsupported state format version {version}"),
    }
}

fn add_legacy_checkpoint_identity(
    mut state: RunState,
    relative_path: Option<String>,
    root: &Path,
) -> Result<RunState> {
    let Some(relative_path) = relative_path
    else {
        return Ok(state);
    };
    let path = root.join(&relative_path);
    if !path.is_file() {
        return Ok(state);
    }

    state.checkpoint_identity = Some(CheckpointIdentity {
        generation: state.model_generation,
        relative_path,
        sha256: crate::artifact::sha256_file(&path)?,
    });

    Ok(state)
}

fn write_model_immutably(
    final_path: &Path,
    write_model: impl FnOnce(&Path) -> Result<()>,
) -> Result<()> {
    let temporary = crate::artifact::unique_sibling(final_path);
    let mut cleanup = crate::artifact::TemporaryPath::new(temporary.clone());

    write_model(&temporary).with_context(|| format!("writing {}", temporary.display()))?;
    File::open(&temporary)?.sync_all()?;

    if final_path.exists() {
        anyhow::ensure!(
            crate::artifact::sha256_file(&temporary)? == crate::artifact::sha256_file(final_path)?,
            "checkpoint generation already exists with different contents: {}",
            final_path.display()
        );

        return Ok(());
    }

    fs::rename(&temporary, final_path)?;
    cleanup.keep();
    crate::artifact::sync_parent(final_path)?;

    Ok(())
}

fn write_json_atomic<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    write_bytes_atomic(path, &serde_json::to_vec_pretty(value)?)?;

    Ok(())
}

fn write_toml_atomic<T: serde::Serialize>(path: &Path, value: &T) -> Result<()> {
    write_bytes_atomic(path, toml::to_string_pretty(value)?.as_bytes())?;

    Ok(())
}

fn write_bytes_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let temporary = crate::artifact::unique_sibling(path);
    let mut cleanup = crate::artifact::TemporaryPath::new(temporary.clone());
    let mut file = File::create(&temporary)?;

    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    cleanup.keep();
    crate::artifact::sync_parent(path)
}

fn copy_atomically(source: &Path, destination: &Path) -> Result<()> {
    let temporary = crate::artifact::unique_sibling(destination);
    let mut cleanup = crate::artifact::TemporaryPath::new(temporary.clone());

    fs::copy(source, &temporary)?;
    File::open(&temporary)?.sync_all()?;
    fs::rename(&temporary, destination)?;
    cleanup.keep();
    crate::artifact::sync_parent(destination)
}

#[cfg(unix)]
fn lock_exclusive(file: &File) -> Result<()> {
    use std::os::fd::AsRawFd;

    let status = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    anyhow::ensure!(status == 0, "run is already locked by another writer");

    Ok(())
}

#[cfg(not(unix))]
fn lock_exclusive(_file: &File) -> Result<()> {
    anyhow::bail!("run writer locking is unsupported on this platform")
}

fn relative_to_root(root: &Path, path: &Path) -> Result<String> {
    Ok(path.strip_prefix(root)?.to_string_lossy().into_owned())
}
