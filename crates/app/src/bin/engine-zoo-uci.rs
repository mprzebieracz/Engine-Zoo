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

fn write_uci_options(out: &mut impl Write, settings: &Settings) -> anyhow::Result<()> {
    writeln!(out, "id name engine-zoo-uci")?;
    writeln!(out, "id author engine-zoo")?;
    writeln!(
        out,
        "option name Model type string default {}",
        settings.model
    )?;
    writeln!(
        out,
        "option name RunDir type string default {}",
        settings.run_dir.display()
    )?;
    writeln!(
        out,
        "option name Simulations type spin default {} min 1 max 1000000",
        settings.simulations
    )?;
    writeln!(out, "option name Device type string default auto")?;
    writeln!(
        out,
        "option name Temperature type spin default 0 min 0 max 100"
    )?;
    writeln!(
        out,
        "option name OpeningPlies type spin default {} min 0 max 100",
        settings.opening_plies
    )?;
    writeln!(
        out,
        "option name Threads type spin default {} min 1 max 256",
        settings.threads
    )?;
    writeln!(out, "option name TensorRtModule type string default")?;
    writeln!(out, "uciok")?;
    Ok(())
}

fn set_option(engine: &mut ChessUciEngine, name: &str, value: String) -> anyhow::Result<()> {
    let name = name.to_ascii_lowercase();
    let invalidates_model =
        matches!(name.as_str(), "model" | "rundir" | "device" | "tensorrtmodule");

    match name.as_str() {
        "model" => engine.settings.model = value,
        "rundir" => engine.settings.run_dir = PathBuf::from(value),
        "simulations" => engine.settings.simulations = value.parse::<usize>()?.max(1),
        "device" => engine.settings.device = device(&value)?,
        "temperature" => engine.settings.temperature = value.parse::<f32>()?,
        "openingplies" => engine.settings.opening_plies = value.parse::<usize>()?,
        "threads" => engine.settings.threads = value.parse::<usize>()?.max(1),
        "tensorrtmodule" => {
            engine.settings.tensor_rt_module = if value.trim().is_empty() {
                None
            }
            else {
                Some(PathBuf::from(value))
            };
        }
        _ => return Ok(()),
    }

    if invalidates_model {
        engine.invalidate_model();
    }

    Ok(())
}

fn go_simulations(
    engine: &ChessUciEngine,
    movetime_ms: Option<u64>,
    nodes: Option<usize>,
) -> Option<usize> {
    nodes.or_else(|| {
        movetime_ms.map(|milliseconds| {
            engine
                .settings
                .simulations
                .max((milliseconds / 10) as usize)
                .max(1)
        })
    })
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
            UciCommand::Uci => write_uci_options(&mut out, &engine.settings)?,
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
                if let Err(e) = set_option(&mut engine, &name, value) {
                    writeln!(out, "info string invalid option {name}: {e}")?;
                }
            }
            UciCommand::Go { movetime_ms, nodes } => {
                match engine.bestmove(go_simulations(&engine, movetime_ms, nodes)) {
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
