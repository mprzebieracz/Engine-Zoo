use clap::Parser;
use engine_app::uci::{parse_command, ChessUciEngine, Settings, UciCommand};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use tch::Device;

#[derive(Parser, Debug)]
#[command(
    name = "engine-zoo-uci",
    about = "Deterministic AlphaZero chess UCI engine"
)]
struct Args {
    #[arg(long)]
    run_dir: Option<PathBuf>,
    #[arg(long)]
    checkpoint: Option<String>,
    #[arg(long, default_value_t = 800)]
    simulations: usize,
    #[arg(long, default_value = "auto")]
    device: String,
    #[arg(long, default_value_t = 1)]
    threads: usize,
}

fn device(s: &str) -> anyhow::Result<Device> {
    match s.to_ascii_lowercase().as_str() {
        "auto" => Ok(Device::cuda_if_available()),
        "cpu" => Ok(Device::Cpu),
        "cuda" | "cuda:0" => {
            anyhow::ensure!(tch::Cuda::is_available(), "CUDA is not available");
            Ok(Device::Cuda(0))
        }
        other => anyhow::bail!("unknown device {other}; use auto, cpu, or cuda"),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut settings = Settings::default();
    if let Some(run_dir) = args.run_dir {
        settings.run_dir = run_dir;
    }
    if let Some(checkpoint) = args.checkpoint {
        settings.model = checkpoint;
    }
    settings.simulations = args.simulations;
    settings.threads = args.threads.max(1);
    settings.device = device(&args.device)?;
    let mut engine = ChessUciEngine::new(settings);
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let command = parse_command(&line?);
        match command {
            UciCommand::Uci => {
                writeln!(out, "id name engine-zoo-uci")?;
                writeln!(out, "id author engine-zoo")?;
                writeln!(
                    out,
                    "option name Model type string default {}",
                    engine.settings.model
                )?;
                writeln!(
                    out,
                    "option name RunDir type string default {}",
                    engine.settings.run_dir.display()
                )?;
                writeln!(
                    out,
                    "option name Simulations type spin default {} min 1 max 1000000",
                    engine.settings.simulations
                )?;
                writeln!(out, "option name Device type string default auto")?;
                writeln!(
                    out,
                    "option name Temperature type spin default 0 min 0 max 100"
                )?;
                writeln!(
                    out,
                    "option name OpeningPlies type spin default {} min 0 max 100",
                    engine.settings.opening_plies
                )?;
                writeln!(
                    out,
                    "option name Threads type spin default {} min 1 max 256",
                    engine.settings.threads
                )?;
                writeln!(out, "uciok")?;
            }
            UciCommand::IsReady => writeln!(out, "readyok")?,
            UciCommand::UciNewGame => {
                engine.new_game();
            }
            UciCommand::Position { fen, moves } => {
                if let Err(e) = engine.set_position(fen.as_deref(), &moves) {
                    writeln!(out, "info string {e}")?;
                }
            }
            UciCommand::SetOption { name, value } => {
                let n = name.to_ascii_lowercase();
                let result = match n.as_str() {
                    "model" => {
                        engine.settings.model = value;
                        Ok(())
                    }
                    "rundir" => {
                        engine.settings.run_dir = PathBuf::from(value);
                        Ok(())
                    }
                    "simulations" => value
                        .parse::<usize>()
                        .map(|v| engine.settings.simulations = v.max(1))
                        .map_err(Into::into),
                    "device" => device(&value).map(|d| engine.settings.device = d),
                    "temperature" => value
                        .parse::<f32>()
                        .map(|v| engine.settings.temperature = v)
                        .map_err(Into::into),
                    "openingplies" => value
                        .parse::<usize>()
                        .map(|v| engine.settings.opening_plies = v)
                        .map_err(Into::into),
                    "threads" => value
                        .parse::<usize>()
                        .map(|v| engine.settings.threads = v.max(1))
                        .map_err(Into::into),
                    _ => Ok(()),
                };
                if result.is_ok() && matches!(n.as_str(), "model" | "rundir" | "device") {
                    engine.invalidate_model();
                }
                if let Err(e) = result {
                    writeln!(out, "info string invalid option {name}: {e}")?;
                }
            }
            UciCommand::Go { movetime_ms, nodes } => {
                let simulations = nodes.or_else(|| {
                    movetime_ms.map(|ms| engine.settings.simulations.max((ms / 10) as usize).max(1))
                });
                match engine.bestmove(simulations) {
                    Ok(mv) => writeln!(out, "bestmove {mv}")?,
                    Err(e) => writeln!(out, "info string {e}\nbestmove 0000")?,
                }
            }
            UciCommand::Stop => {}
            UciCommand::Quit => break,
            UciCommand::Unknown => {}
        }
        out.flush()?;
    }
    Ok(())
}
