use super::network::{ChessAzV2Config, NetConfig, NetworkConfig};
use super::representation::{ChessV1Representation, Connect4AzRepresentation};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const RUN_CONFIG_FORMAT_VERSION: u32 = 2;

/// Scalar AlphaZero hyperparameters for Connect4. Representation dimensions are fixed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Connect4ScalarAzConfig {
    pub num_res_blocks: i64,
    pub num_filters: i64,
}

impl Connect4ScalarAzConfig {
    fn network_config(&self) -> NetConfig {
        NetConfig::for_representation::<games::Connect4, Connect4AzRepresentation>(
            self.num_res_blocks,
            self.num_filters,
        )
    }

    fn from_legacy(net: NetConfig) -> Result<Self> {
        validate_legacy_shape(
            "Connect4 scalar AlphaZero",
            &net,
            &Self {
                num_res_blocks: net.num_res_blocks,
                num_filters: net.num_filters,
            }
            .network_config(),
        )?;
        Ok(Self {
            num_res_blocks: net.num_res_blocks,
            num_filters: net.num_filters,
        })
    }
}

/// Scalar AlphaZero hyperparameters for Chess v1. Representation dimensions are fixed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChessScalarAzV1Config {
    pub num_res_blocks: i64,
    pub num_filters: i64,
}

impl ChessScalarAzV1Config {
    fn network_config(&self) -> NetConfig {
        NetConfig::for_representation::<games::ChessPosition, ChessV1Representation>(
            self.num_res_blocks,
            self.num_filters,
        )
    }

    fn from_legacy(net: NetConfig) -> Result<Self> {
        validate_legacy_shape(
            "Chess scalar AlphaZero v1",
            &net,
            &Self {
                num_res_blocks: net.num_res_blocks,
                num_filters: net.num_filters,
            }
            .network_config(),
        )?;
        Ok(Self {
            num_res_blocks: net.num_res_blocks,
            num_filters: net.num_filters,
        })
    }
}

fn validate_legacy_shape(model: &str, found: &NetConfig, expected: &NetConfig) -> Result<()> {
    anyhow::ensure!(
        found.input_channels == expected.input_channels
            && found.height == expected.height
            && found.width == expected.width
            && found.action_size == expected.action_size,
        "legacy {model} config has incompatible representation dimensions: expected [{}, {}, {}] with {} actions, got [{}, {}, {}] with {} actions",
        expected.input_channels,
        expected.height,
        expected.width,
        expected.action_size,
        found.input_channels,
        found.height,
        found.width,
        found.action_size,
    );
    Ok(())
}

/// The model fully determines a run's game, representation, and network.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "kebab-case")]
pub enum ModelConfig {
    Connect4ScalarAz(Connect4ScalarAzConfig),
    ChessScalarAzV1(ChessScalarAzV1Config),
    ChessAzV2(ChessAzV2Config),
}

impl ModelConfig {
    pub fn game_name(&self) -> &'static str {
        match self {
            Self::Connect4ScalarAz(_) => "connect4",
            Self::ChessScalarAzV1(_) | Self::ChessAzV2(_) => "chess",
        }
    }

    pub fn network_config(&self) -> NetworkConfig {
        match self {
            Self::Connect4ScalarAz(cfg) => NetworkConfig::Legacy(cfg.network_config()),
            Self::ChessScalarAzV1(cfg) => NetworkConfig::Legacy(cfg.network_config()),
            Self::ChessAzV2(cfg) => NetworkConfig::ChessAzV2(*cfg),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunConfig {
    pub format_version: u32,
    pub model: ModelConfig,
}

#[derive(Deserialize)]
struct LegacyRunConfig {
    game: String,
    net: NetConfig,
    #[serde(default)]
    architecture: LegacyRunArchitecture,
}

#[derive(Default, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "kebab-case")]
enum LegacyRunArchitecture {
    #[default]
    Legacy,
    ChessAzV2(ChessAzV2Config),
}

impl RunConfig {
    /// Parses current versioned configs and the previous `{ game, net, architecture }` format.
    pub fn parse_json(json: &str) -> Result<Self> {
        let cfg = match serde_json::from_str(json) {
            Ok(cfg) => cfg,
            Err(current_error) => {
                let legacy: LegacyRunConfig = serde_json::from_str(json).with_context(|| {
                    format!("not a versioned run config ({current_error}); trying legacy config")
                })?;
                let model = match (legacy.game.as_str(), legacy.architecture) {
                    ("connect4", LegacyRunArchitecture::Legacy) => ModelConfig::Connect4ScalarAz(
                        Connect4ScalarAzConfig::from_legacy(legacy.net)?,
                    ),
                    ("chess", LegacyRunArchitecture::Legacy) => ModelConfig::ChessScalarAzV1(
                        ChessScalarAzV1Config::from_legacy(legacy.net)?,
                    ),
                    ("chess", LegacyRunArchitecture::ChessAzV2(cfg)) => ModelConfig::ChessAzV2(cfg),
                    (game, LegacyRunArchitecture::ChessAzV2(_)) => {
                        anyhow::bail!("chess-az-v2 is only valid for chess runs, not {game}")
                    }
                    (game, LegacyRunArchitecture::Legacy) => {
                        anyhow::bail!("unsupported legacy game {game}")
                    }
                };
                Self {
                    format_version: RUN_CONFIG_FORMAT_VERSION,
                    model,
                }
            }
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn network_config(&self) -> NetworkConfig {
        self.model.network_config()
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            self.format_version == RUN_CONFIG_FORMAT_VERSION,
            "unsupported run config format version {}",
            self.format_version
        );
        if let ModelConfig::ChessAzV2(cfg) = self.model {
            cfg.validate()?;
        }
        Ok(())
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
            RunConfig::parse_json(&fs::read_to_string(&cfg_path)?)
                .with_context(|| format!("parsing {}", cfg_path.display()))?
        } else {
            let cfg = make_config();
            cfg.validate()?;
            fs::create_dir_all(run.checkpoints_dir())?;
            let tmp_path = run.root.join("config.json.tmp");
            fs::write(&tmp_path, serde_json::to_string_pretty(&cfg)?)?;
            fs::rename(tmp_path, &cfg_path)?;
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
        } else {
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
    use crate::representation::{ChessV1Representation, Connect4AzRepresentation};
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
            format_version: RUN_CONFIG_FORMAT_VERSION,
            model: ModelConfig::Connect4ScalarAz(Connect4ScalarAzConfig {
                num_res_blocks: 1,
                num_filters: 8,
            }),
        }
    }

    #[test]
    fn parses_legacy_config_as_a_model_variant() {
        let cfg = RunConfig::parse_json(
            r#"{"game":"chess","net":{"input_channels":1,"height":1,"width":1,"num_res_blocks":1,"num_filters":1,"action_size":1},"architecture":{"kind":"chess-az-v2","config":{"history":4}}}"#,
        )
        .unwrap();
        assert!(matches!(
            cfg.model,
            ModelConfig::ChessAzV2(ChessAzV2Config { history: 4 })
        ));
    }

    #[test]
    fn converts_legacy_scalar_configs_after_validating_their_shapes() {
        let connect4 =
            NetConfig::for_representation::<games::Connect4, Connect4AzRepresentation>(1, 8);
        let chess =
            NetConfig::for_representation::<games::ChessPosition, ChessV1Representation>(2, 16);
        for (game, net) in [("connect4", connect4), ("chess", chess)] {
            let cfg =
                RunConfig::parse_json(&serde_json::json!({ "game": game, "net": net }).to_string())
                    .unwrap();
            assert!(matches!(
                cfg.model,
                ModelConfig::Connect4ScalarAz(_) | ModelConfig::ChessScalarAzV1(_)
            ));
        }
    }

    #[test]
    fn rejects_legacy_scalar_config_with_wrong_shape() {
        let err = RunConfig::parse_json(
            r#"{"game":"connect4","net":{"input_channels":1,"height":1,"width":1,"num_res_blocks":1,"num_filters":1,"action_size":1}}"#,
        )
        .unwrap_err();
        assert!(err
            .to_string()
            .contains("incompatible representation dimensions"));
    }

    #[test]
    fn new_runs_write_the_versioned_model_format() {
        let root = test_root("versioned-config");
        let _ = RunDir::open_or_create(&root, config).unwrap();
        let json: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(root.join("config.json")).unwrap()).unwrap();
        assert_eq!(json["format_version"], RUN_CONFIG_FORMAT_VERSION);
        assert_eq!(json["model"]["kind"], "connect4-scalar-az");
        assert!(json["model"]["config"].get("input_channels").is_none());
        assert!(json.get("game").is_none());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn invalid_new_runs_leave_no_config_or_checkpoint_directory() {
        let root = test_root("invalid-new-run");
        let err = RunDir::open_or_create(&root, || RunConfig {
            format_version: 0,
            model: config().model,
        });

        assert!(err.is_err());
        assert!(!root.join("config.json").exists());
        assert!(!root.join("checkpoints").exists());
        assert!(!root.exists());
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
