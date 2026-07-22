use super::{ExperimentConfig, ReplayConfig, EXPERIMENT_FORMAT_VERSION};
use crate::{ChessHistoryLength, ModelSpec, SelfPlayConfig, TrainConfig, ValueHeadConfig};
use anyhow::{Context, Result};
use serde::Deserialize;

/// Reads the former model-only `config.json` format into a complete default
/// experiment. New writes always use `experiment.json`.
pub(super) fn migrate_old_config(json: &str) -> Result<ExperimentConfig> {
    #[derive(Deserialize)]
    struct OldRunConfig {
        format_version: u32,
        model: ModelSpec,
    }
    let model = match serde_json::from_str::<OldRunConfig>(json) {
        Ok(old) => {
            anyhow::ensure!(
                old.format_version == 3,
                "unsupported old run config format version {}",
                old.format_version
            );
            old.model
        }
        Err(_) => migrate_historical_model(json)?,
    };
    let self_play = SelfPlayConfig::default();
    Ok(ExperimentConfig {
        format_version: EXPERIMENT_FORMAT_VERSION,
        model,
        self_play,
        replay: ReplayConfig::default(),
        training: TrainConfig::default(),
        seed: 0,
    })
}

#[derive(Deserialize)]
struct HistoricalVersionedRun {
    format_version: u32,
    model: HistoricalModel,
}

#[derive(Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "kebab-case")]
enum HistoricalModel {
    Connect4ScalarAz(ScalarConfig),
    ChessScalarAzV1(ScalarConfig),
    ChessAzV2 { history: usize },
}

#[derive(Deserialize)]
struct ScalarConfig {
    num_res_blocks: i64,
    num_filters: i64,
}

#[derive(Deserialize)]
struct HistoricalFlatRun {
    game: String,
    net: HistoricalNet,
    #[serde(default)]
    architecture: HistoricalArchitecture,
}

#[derive(Deserialize)]
struct HistoricalNet {
    input_channels: i64,
    height: i64,
    width: i64,
    num_res_blocks: i64,
    num_filters: i64,
    action_size: i64,
}

#[derive(Default, Deserialize)]
#[serde(tag = "kind", content = "config", rename_all = "kebab-case")]
enum HistoricalArchitecture {
    #[default]
    Legacy,
    ChessAzV2 {
        history: usize,
    },
}

fn migrate_historical_model(json: &str) -> Result<ModelSpec> {
    if let Ok(run) = serde_json::from_str::<HistoricalVersionedRun>(json) {
        anyhow::ensure!(
            run.format_version == 2,
            "unsupported historical run config format version {}",
            run.format_version
        );
        return match run.model {
            HistoricalModel::Connect4ScalarAz(config) => classic_connect4(config),
            HistoricalModel::ChessScalarAzV1(config) => classic_chess(config),
            HistoricalModel::ChessAzV2 { history } => canonical_chess(history),
        };
    }

    let run: HistoricalFlatRun =
        serde_json::from_str(json).context("parsing historical run config")?;
    match (run.game.as_str(), run.architecture) {
        ("connect4", HistoricalArchitecture::Legacy) => {
            validate_shape(&run.net, [1, 6, 7], 7, "Connect4 scalar AlphaZero")?;
            classic_connect4(ScalarConfig {
                num_res_blocks: run.net.num_res_blocks,
                num_filters: run.net.num_filters,
            })
        }
        ("chess", HistoricalArchitecture::Legacy) => {
            validate_shape(&run.net, [19, 8, 8], 20_480, "classic chess AlphaZero")?;
            classic_chess(ScalarConfig {
                num_res_blocks: run.net.num_res_blocks,
                num_filters: run.net.num_filters,
            })
        }
        ("chess", HistoricalArchitecture::ChessAzV2 { history }) => canonical_chess(history),
        (game, HistoricalArchitecture::ChessAzV2 { .. }) => {
            anyhow::bail!("chess-az-v2 is only valid for chess runs, not {game}")
        }
        (game, HistoricalArchitecture::Legacy) => {
            anyhow::bail!("unsupported historical game {game}")
        }
    }
}

fn classic_connect4(config: ScalarConfig) -> Result<ModelSpec> {
    Ok(ModelSpec::connect4_basic(
        positive_blocks(config.num_res_blocks)?,
        positive_channels(config.num_filters)?,
    ))
}

fn classic_chess(config: ScalarConfig) -> Result<ModelSpec> {
    Ok(ModelSpec::chess_classic(
        positive_blocks(config.num_res_blocks)?,
        positive_channels(config.num_filters)?,
    ))
}

fn canonical_chess(history: usize) -> Result<ModelSpec> {
    let history = match history {
        1 => ChessHistoryLength::One,
        4 => ChessHistoryLength::Four,
        8 => ChessHistoryLength::Eight,
        _ => anyhow::bail!("unsupported chess canonical history length {history}"),
    };
    Ok(ModelSpec::chess_se(
        history,
        ValueHeadConfig::Wdl { hidden: 128 },
    ))
}

fn validate_shape(
    found: &HistoricalNet,
    expected: [i64; 3],
    action_size: i64,
    model: &str,
) -> Result<()> {
    anyhow::ensure!(
        [found.input_channels, found.height, found.width] == expected
            && found.action_size == action_size,
        "historical {model} config has incompatible representation dimensions: expected {:?} with {action_size} actions, got [{}, {}, {}] with {} actions",
        expected,
        found.input_channels,
        found.height,
        found.width,
        found.action_size,
    );
    Ok(())
}

fn positive_blocks(value: i64) -> Result<usize> {
    let value = usize::try_from(value).context("residual block count must be positive")?;
    anyhow::ensure!(value > 0, "residual block count must be positive");
    Ok(value)
}

fn positive_channels(value: i64) -> Result<i64> {
    anyhow::ensure!(value > 0, "residual channel count must be positive");
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        NetworkSpec, PolicyHeadConfig, RepresentationSpec, ResidualTrunkConfig, ValueHeadConfig,
    };

    #[test]
    fn migrates_the_historical_classic_chess_model_without_changing_its_shape() {
        let config = migrate_old_config(
            r#"{
                "game":"chess",
                "net":{
                    "input_channels":19,
                    "height":8,
                    "width":8,
                    "num_res_blocks":10,
                    "num_filters":64,
                    "action_size":20480
                }
            }"#,
        )
        .unwrap();
        assert_eq!(
            config.model.representation,
            RepresentationSpec::ChessClassic
        );
        let NetworkSpec::Residual(network) = config.model.network;
        assert_eq!(
            network.trunk,
            ResidualTrunkConfig::Basic {
                blocks: 10,
                channels: 64,
            }
        );
        assert_eq!(network.policy_head, PolicyHeadConfig::Dense { channels: 2 });
        assert_eq!(network.value_head, ValueHeadConfig::Scalar { hidden: 64 });
    }

    #[test]
    fn rejects_historical_chess_with_a_wrong_policy_size() {
        let error = migrate_old_config(
            r#"{
                "game":"chess",
                "net":{
                    "input_channels":19,
                    "height":8,
                    "width":8,
                    "num_res_blocks":10,
                    "num_filters":64,
                    "action_size":4672
                }
            }"#,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("incompatible representation dimensions"));
    }
}
