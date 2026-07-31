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
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

/// Inference backend used by an [`ArenaModel`] when it is loaded through the
/// UCI adapter. The default is [`ArenaBackend::Native`] so that existing
/// arenas—and their tests—do not silently start requiring TensorRT tooling.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ArenaBackend {
    #[default]
    Native,
    TensorRt,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ArenaModel {
    pub name: String,
    pub path: PathBuf,
    pub architecture: String,
    #[serde(default)]
    pub simulations: Option<usize>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub backend: ArenaBackend,
    /// Precompiled Torch-TensorRT TorchScript module for this checkpoint. When
    /// `backend` is `tensor-rt` and this is `None`, the caller (typically the
    /// Python arena wrapper) is expected to compile the module into the
    /// configured cache directory and populate this field before Rust runs
    /// matches. Providing it directly for a `native` model is not allowed.
    #[serde(default)]
    pub tensor_rt_module: Option<PathBuf>,
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
    /// Python interpreter used by the Torch-TensorRT compiler. Only consumed
    /// by the Python arena wrapper; the Rust binary carries it forward so
    /// that `arena.toml` is the single source of truth.
    #[serde(default)]
    pub tensor_rt_python: Option<PathBuf>,
    #[serde(default = "default_trt_cache")]
    pub trt_cache_dir: PathBuf,
    #[serde(default = "default_trt_opt")]
    pub trt_opt_batch_size: usize,
    #[serde(default = "default_trt_max")]
    pub trt_max_batch_size: usize,
    #[serde(default = "default_trt_min")]
    pub trt_min_batch_size: usize,
    #[serde(default = "default_trt_compiler")]
    pub tensor_rt_compiler: PathBuf,
    #[serde(default = "default_model_cli")]
    pub model_cli: PathBuf,
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
            tensor_rt_python: None,
            trt_cache_dir: default_trt_cache(),
            trt_opt_batch_size: default_trt_opt(),
            trt_max_batch_size: default_trt_max(),
            trt_min_batch_size: default_trt_min(),
            tensor_rt_compiler: default_trt_compiler(),
            model_cli: default_model_cli(),
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
    pub backend: ArenaBackend,
    pub simulations: usize,
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
            backend: opponent.backend,
            simulations: opponent.simulations.unwrap_or(settings.simulations),
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
    fs::write(
        output_root.join("REPORT.md"),
        render_report(&summary, config),
    )?;
    Ok(summary)
}

pub(crate) fn model_engine(model: &ArenaModel, config: &ArenaConfig) -> Engine {
    let mut engine = Engine::new(&config.settings.uci, &model.name)
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
        .option("OpeningPlies", config.settings.opening_plies.to_string());

    // Set TensorRtModule when we have a compiled module. UCI treats an empty
    // value as "force native", so we only advertise the option when it has a
    // concrete path.
    if let Some(module) = &model.tensor_rt_module {
        engine = engine.option("TensorRtModule", module.display().to_string());
    }
    engine
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
        if let Some(module) = &mut model.tensor_rt_module {
            if module.is_relative() {
                *module = root.join(&*module);
            }
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
    if config.settings.trt_cache_dir.is_relative() {
        config.settings.trt_cache_dir = root.join(&config.settings.trt_cache_dir);
    }
    if let Some(path) = &mut config.settings.opening_file {
        if path.is_relative() {
            *path = root.join(&*path);
        }
    }
    if should_resolve(&config.settings.tensor_rt_compiler) {
        config.settings.tensor_rt_compiler = root.join(&config.settings.tensor_rt_compiler);
    }
    if should_resolve(&config.settings.model_cli) {
        config.settings.model_cli = root.join(&config.settings.model_cli);
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
        if model.backend == ArenaBackend::Native && model.tensor_rt_module.is_some() {
            bail!(
                "model {} has backend=native but a tensor_rt_module path is set",
                model.name
            );
        }
    }
    Ok(())
}

fn render_report(summary: &ArenaSummary, config: &ArenaConfig) -> String {
    let mut out = String::new();
    let settings = &config.settings;
    let candidate_sims = summary
        .candidate
        .simulations
        .unwrap_or(settings.simulations);
    let device = summary
        .candidate
        .device
        .clone()
        .unwrap_or_else(|| settings.device.clone());

    let _ = writeln!(out, "# Arena report: {}", summary.candidate.name);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "- candidate architecture: `{}`",
        summary.candidate.architecture
    );
    let _ = writeln!(
        out,
        "- candidate checkpoint: `{}`",
        summary.candidate.path.display()
    );
    let _ = writeln!(
        out,
        "- candidate backend: `{}`",
        backend_label(summary.candidate.backend)
    );
    let _ = writeln!(out, "- candidate simulations: {candidate_sims}");
    let _ = writeln!(out, "- device: `{device}`");
    let _ = writeln!(
        out,
        "- games per opponent: {} ({} total)",
        summary.games, summary.total_games
    );
    let _ = writeln!(out, "- time control: `{}`", settings.time_control);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "Aggregate score {:.1}% ({}W / {}D / {}L), smoothed Elo {:+.1}.",
        summary.score_fraction * 100.0,
        summary.total_score.wins,
        summary.total_score.draws,
        summary.total_score.losses,
        summary.elo_delta
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| Opponent | Backend | Sims | W | D | L | Score % | Elo Δ |"
    );
    let _ = writeln!(
        out,
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for opponent in &summary.opponents {
        let _ = writeln!(
            out,
            "| `{}` | {} | {} | {} | {} | {} | {:.1}% | {:+.1} |",
            opponent.name,
            backend_label(opponent.backend),
            opponent.simulations,
            opponent.score.wins,
            opponent.score.draws,
            opponent.score.losses,
            opponent.score_fraction * 100.0,
            opponent.elo_delta
        );
    }
    out
}

fn backend_label(backend: ArenaBackend) -> &'static str {
    match backend {
        ArenaBackend::Native => "native",
        ArenaBackend::TensorRt => "tensor-rt",
    }
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
fn default_trt_cache() -> PathBuf {
    PathBuf::from("data/evaluations/trt-cache")
}
fn default_trt_min() -> usize {
    1
}
fn default_trt_opt() -> usize {
    32
}
fn default_trt_max() -> usize {
    256
}
fn default_trt_compiler() -> PathBuf {
    PathBuf::from("scripts/compile_tensorrt.py")
}
fn default_model_cli() -> PathBuf {
    PathBuf::from("target/release/engine-zoo-model")
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
        assert_eq!(config.candidate.backend, ArenaBackend::Native);
        assert!(config.candidate.tensor_rt_module.is_none());
    }

    #[test]
    fn parses_tensor_rt_backend_and_optional_module() {
        let config: ArenaConfig = toml::from_str(
            r#"
            [candidate]
            name = "trt"
            path = "models/trt.safetensors"
            architecture = "chess-se-h4"
            backend = "tensor-rt"
            tensor_rt_module = "cache/trt.ts"

            [[opponents]]
            name = "native"
            path = "models/native.safetensors"
            architecture = "chess-se"

            [settings]
            games = 2
            simulations = 100
        "#,
        )
        .unwrap();
        assert_eq!(config.candidate.backend, ArenaBackend::TensorRt);
        assert_eq!(
            config.candidate.tensor_rt_module.as_deref(),
            Some(Path::new("cache/trt.ts"))
        );
        assert_eq!(config.opponents[0].backend, ArenaBackend::Native);
    }

    #[test]
    fn rejects_odd_game_counts_and_duplicate_names() {
        let model = ArenaModel {
            name: "same".into(),
            path: "x".into(),
            architecture: "a".into(),
            simulations: None,
            device: None,
            backend: ArenaBackend::Native,
            tensor_rt_module: None,
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

    #[test]
    fn rejects_tensor_rt_module_on_native_backend() {
        let candidate = ArenaModel {
            name: "candidate".into(),
            path: "candidate.safetensors".into(),
            architecture: "chess-se".into(),
            simulations: None,
            device: None,
            backend: ArenaBackend::Native,
            tensor_rt_module: Some(PathBuf::from("trt.ts")),
        };
        let opponent = ArenaModel {
            name: "opponent".into(),
            path: "opponent.safetensors".into(),
            architecture: "chess-se".into(),
            simulations: None,
            device: None,
            backend: ArenaBackend::Native,
            tensor_rt_module: None,
        };
        let config = ArenaConfig {
            candidate,
            opponents: vec![opponent],
            settings: ArenaSettings {
                games: 2,
                ..Default::default()
            },
        };
        assert!(validate(&config).is_err());
    }

    #[test]
    fn model_engine_includes_tensor_rt_module_option_when_set() {
        let config = ArenaConfig {
            candidate: ArenaModel {
                name: "candidate".into(),
                path: PathBuf::from("candidate.safetensors"),
                architecture: "chess-se".into(),
                simulations: None,
                device: None,
                backend: ArenaBackend::TensorRt,
                tensor_rt_module: Some(PathBuf::from("cache/candidate.trt.ts")),
            },
            opponents: vec![ArenaModel {
                name: "baseline".into(),
                path: PathBuf::from("baseline.safetensors"),
                architecture: "chess-se".into(),
                simulations: None,
                device: None,
                backend: ArenaBackend::Native,
                tensor_rt_module: None,
            }],
            settings: ArenaSettings {
                games: 2,
                ..Default::default()
            },
        };

        let candidate = model_engine(&config.candidate, &config);
        assert!(candidate
            .options
            .iter()
            .any(|(key, value)| key == "TensorRtModule" && value == "cache/candidate.trt.ts"));

        let opponent = model_engine(&config.opponents[0], &config);
        assert!(!opponent
            .options
            .iter()
            .any(|(key, _)| key == "TensorRtModule"));
    }
}
