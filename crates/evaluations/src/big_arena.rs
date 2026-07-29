//! Config-driven multi-opponent chess arenas.
//!
//! The arena deliberately delegates each pair to [`match_suite::run_match`].
//! This keeps PGN cleaning, Fastchess diagnostics, and report persistence
//! identical to one-off evaluations while adding aggregate tournament output.

use crate::fastchess::{Engine, FastchessCommand, Openings};
use crate::match_suite::run_match;
use crate::model_engine::infer_run_dir;
use crate::report::{EvaluationSpec, Score, SearchSpec, SCHEMA_VERSION};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArenaModel {
    pub name: String,
    pub path: PathBuf,
    pub architecture: String,
    #[serde(default)]
    pub simulations: Option<usize>,
    #[serde(default)]
    pub device: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArenaSettings {
    #[serde(default = "default_fastchess")]
    pub fastchess: PathBuf,
    #[serde(default = "default_uci")]
    pub uci: PathBuf,
    #[serde(default = "default_output")]
    pub output_dir: PathBuf,
    #[serde(default = "default_games")]
    pub games: u32,
    #[serde(default = "default_simulations")]
    pub simulations: usize,
    #[serde(default = "default_device")]
    pub device: String,
    #[serde(default = "default_concurrency")]
    pub concurrency: u32,
    #[serde(default = "default_tc")]
    pub time_control: String,
    #[serde(default = "default_max_moves")]
    pub max_moves: u32,
    #[serde(default)]
    pub opening_file: Option<PathBuf>,
    #[serde(default)]
    pub opening_plies: u32,
}

impl Default for ArenaSettings {
    fn default() -> Self {
        Self {
            fastchess: default_fastchess(),
            uci: default_uci(),
            output_dir: default_output(),
            games: default_games(),
            simulations: default_simulations(),
            device: default_device(),
            concurrency: default_concurrency(),
            time_control: default_tc(),
            max_moves: default_max_moves(),
            opening_file: None,
            opening_plies: 0,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArenaConfig {
    pub candidate: ArenaModel,
    pub opponents: Vec<ArenaModel>,
    #[serde(default)]
    pub settings: ArenaSettings,
}

#[derive(Clone, Debug, Serialize)]
pub struct OpponentResult {
    pub name: String,
    pub architecture: String,
    pub report: String,
    pub score: Score,
    pub score_fraction: f64,
    pub elo_delta: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArenaSummary {
    pub candidate: ArenaModel,
    pub opponents: Vec<OpponentResult>,
    pub games: u32,
    pub total_games: u32,
    pub total_score: Score,
    pub score_fraction: f64,
    pub elo_delta: f64,
}

pub fn load_config(path: &Path) -> Result<ArenaConfig> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("reading arena config {}", path.display()))?;
    let mut config: ArenaConfig = toml::from_str(&text)
        .with_context(|| format!("parsing arena config {}", path.display()))?;
    let root = path.parent().unwrap_or_else(|| Path::new("."));
    resolve_paths(&mut config, root);
    validate(&config)?;
    Ok(config)
}

pub fn run_from_config(path: &Path) -> Result<ArenaSummary> {
    let config = load_config(path)?;
    run(&config)
}

pub fn run(config: &ArenaConfig) -> Result<ArenaSummary> {
    validate(config)?;
    let settings = &config.settings;
    let rounds = settings.games / 2;
    let output_root = &settings.output_dir;
    fs::create_dir_all(output_root)?;
    let mut results = Vec::with_capacity(config.opponents.len());

    for opponent in &config.opponents {
        let pair_dir = output_root.join(&opponent.name);
        let candidate = model_engine(&config.candidate, config);
        let baseline = model_engine(opponent, config);
        let command = build_command(
            settings,
            candidate.clone(),
            baseline.clone(),
            rounds,
            &pair_dir,
        );
        let spec = build_spec(&config.candidate, opponent, settings, candidate, baseline);
        let report = run_match(&pair_dir, spec, command)
            .with_context(|| format!("evaluating candidate against {}", opponent.name))?;
        results.push(OpponentResult {
            name: opponent.name.clone(),
            architecture: opponent.architecture.clone(),
            report: format!("{}/report.json", opponent.name),
            score: report.score,
            score_fraction: report.score_fraction,
            elo_delta: report.smoothed_elo_delta,
        });
    }

    let total_score = results.iter().fold(Score::default(), |mut total, result| {
        total.wins += result.score.wins;
        total.draws += result.score.draws;
        total.losses += result.score.losses;
        total
    });
    let summary = ArenaSummary {
        candidate: config.candidate.clone(),
        opponents: results,
        games: settings.games,
        total_games: settings.games * config.opponents.len() as u32,
        score_fraction: total_score.fraction(),
        elo_delta: total_score.smoothed_elo_delta(),
        total_score,
    };
    fs::write(
        output_root.join("arena.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}

fn model_engine(model: &ArenaModel, config: &ArenaConfig) -> Engine {
    Engine::new(&config.settings.uci, &model.name)
        .option("RunDir", infer_run_dir(&model.path).display().to_string())
        .option("Model", model.path.display().to_string())
        .option(
            "Simulations",
            model
                .simulations
                .unwrap_or(config.settings.simulations)
                .to_string(),
        )
        .option(
            "Device",
            model.device.as_deref().unwrap_or(&config.settings.device),
        )
        .option("Temperature", "0")
        .option("OpeningPlies", config.settings.opening_plies.to_string())
}

fn build_command(
    settings: &ArenaSettings,
    candidate: Engine,
    baseline: Engine,
    rounds: u32,
    output_dir: &Path,
) -> FastchessCommand {
    let mut command = FastchessCommand::new(&settings.fastchess, candidate, baseline)
        .time_control(&settings.time_control)
        .rounds(rounds)
        .concurrency(settings.concurrency)
        .pgn_output(output_dir.join("games.pgn"))
        .arg("-maxmoves")
        .arg(settings.max_moves.to_string());
    if let Some(opening_file) = &settings.opening_file {
        command = command.openings(Openings::epd(opening_file).plies(settings.opening_plies));
    }
    command
}

fn build_spec(
    candidate_model: &ArenaModel,
    opponent_model: &ArenaModel,
    settings: &ArenaSettings,
    candidate: Engine,
    baseline: Engine,
) -> EvaluationSpec {
    EvaluationSpec {
        schema_version: SCHEMA_VERSION,
        id: format!("arena-{}-vs-{}", candidate_model.name, opponent_model.name),
        suite: "arena".into(),
        candidate: engine_spec(candidate.clone(), candidate_model),
        opponent: Some(engine_spec(baseline.clone(), opponent_model)),
        search: SearchSpec {
            simulations: candidate_model.simulations.unwrap_or(settings.simulations),
            temperature: 0.0,
            dirichlet_noise: false,
            device: candidate_model
                .device
                .clone()
                .unwrap_or_else(|| settings.device.clone()),
        },
        opening_set: settings
            .opening_file
            .as_ref()
            .map(|path| path.display().to_string()),
        seed: None,
        concurrency: settings.concurrency as usize,
        fastchess_version: None,
    }
}

fn engine_spec(engine: Engine, model: &ArenaModel) -> crate::report::EngineSpec {
    crate::report::EngineSpec {
        name: engine.name,
        command: engine.command.display().to_string(),
        args: Vec::new(),
        checkpoint: Some(model.path.display().to_string()),
        options: engine.options,
    }
}

fn resolve_paths(config: &mut ArenaConfig, root: &Path) {
    for model in std::iter::once(&mut config.candidate).chain(config.opponents.iter_mut()) {
        if model.path.is_relative() {
            model.path = root.join(&model.path);
        }
    }
    if should_resolve(&config.settings.fastchess) {
        config.settings.fastchess = root.join(&config.settings.fastchess);
    }
    if should_resolve(&config.settings.uci) {
        config.settings.uci = root.join(&config.settings.uci);
    }
    if config.settings.output_dir.is_relative() {
        config.settings.output_dir = root.join(&config.settings.output_dir);
    }
    if let Some(path) = &mut config.settings.opening_file {
        if path.is_relative() {
            *path = root.join(&*path);
        }
    }
}

/// Bare executable names are resolved through `PATH`; paths containing a
/// directory component are relative to the configuration file.
fn should_resolve(path: &Path) -> bool {
    path.is_relative() && path.parent().is_some_and(|parent| parent != Path::new("."))
}

fn validate(config: &ArenaConfig) -> Result<()> {
    if config.opponents.is_empty() {
        bail!("arena must contain at least one opponent");
    }
    if config.settings.games == 0 || config.settings.games % 2 != 0 {
        bail!("settings.games must be a positive even number");
    }
    if config.settings.simulations == 0
        || config.settings.concurrency == 0
        || config.settings.max_moves == 0
    {
        bail!("simulations, concurrency, and max_moves must be positive");
    }
    let mut names = std::collections::HashSet::new();
    for model in std::iter::once(&config.candidate).chain(config.opponents.iter()) {
        if model.name.trim().is_empty()
            || model.architecture.trim().is_empty()
            || model.path.as_os_str().is_empty()
        {
            bail!("model name, architecture, and path are required");
        }
        if !names.insert(&model.name) {
            bail!("duplicate model name {}", model.name);
        }
    }
    Ok(())
}

fn default_fastchess() -> PathBuf {
    PathBuf::from("fastchess")
}
fn default_uci() -> PathBuf {
    PathBuf::from("target/release/engine-zoo-uci")
}
fn default_output() -> PathBuf {
    PathBuf::from("data/evaluations/arena")
}
fn default_games() -> u32 {
    20
}
fn default_simulations() -> usize {
    800
}
fn default_device() -> String {
    "cpu".into()
}
fn default_concurrency() -> u32 {
    1
}
fn default_tc() -> String {
    "1000000+0".into()
}
fn default_max_moves() -> u32 {
    512
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_validates_multi_model_config() {
        let config: ArenaConfig = toml::from_str(
            r#"
            [candidate]
            name = "new"
            path = "models/new.safetensors"
            architecture = "chess-se"

            [[opponents]]
            name = "old"
            path = "models/old.safetensors"
            architecture = "chess-se"

            [settings]
            games = 4
            simulations = 200
        "#,
        )
        .unwrap();
        validate(&config).unwrap();
        assert_eq!(config.settings.games, 4);
        assert_eq!(config.opponents[0].name, "old");
    }

    #[test]
    fn rejects_odd_game_counts_and_duplicate_names() {
        let model = ArenaModel {
            name: "same".into(),
            path: "x".into(),
            architecture: "a".into(),
            simulations: None,
            device: None,
        };
        let config = ArenaConfig {
            candidate: model.clone(),
            opponents: vec![model],
            settings: ArenaSettings {
                games: 3,
                ..Default::default()
            },
        };
        assert!(validate(&config).is_err());
    }
}
