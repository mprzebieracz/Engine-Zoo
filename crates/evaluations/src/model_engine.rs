//! Shared configuration for exposing saved models through the UCI adapter.

use crate::fastchess::Engine;
use crate::report::EngineSpec;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug)]
pub struct ModelEngine<'a> {
    pub name: &'a str,
    pub uci: &'a Path,
    pub run_dir: &'a Path,
    pub checkpoint: &'a Path,
    pub simulations: usize,
    pub device: &'a str,
    pub opening_plies: Option<u32>,
}

impl ModelEngine<'_> {
    pub fn fastchess(self) -> Engine {
        let mut engine = Engine::new(self.uci, self.name)
            .option("RunDir", self.run_dir.display().to_string())
            .option("Model", self.checkpoint.display().to_string())
            .option("Simulations", self.simulations.to_string())
            .option("Device", self.device)
            .option("Temperature", "0");
        if let Some(plies) = self.opening_plies {
            engine = engine.option("OpeningPlies", plies.to_string());
        }
        engine
    }

    pub fn spec(self) -> EngineSpec {
        let engine = self.fastchess();
        EngineSpec {
            name: engine.name,
            command: self.uci.display().to_string(),
            args: Vec::new(),
            checkpoint: Some(self.checkpoint.display().to_string()),
            options: engine.options,
        }
    }
}

pub fn infer_run_dir(checkpoint: &Path) -> PathBuf {
    checkpoint
        .parent()
        .filter(|parent| parent.file_name().is_some_and(|name| name == "checkpoints"))
        .and_then(Path::parent)
        .unwrap_or_else(|| checkpoint.parent().unwrap_or_else(|| Path::new(".")))
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_run_directory_from_checkpoint_layout() {
        assert_eq!(
            infer_run_dir(Path::new("runs/chess/checkpoints/ckpt_0010.safetensors")),
            Path::new("runs/chess")
        );
        assert_eq!(
            infer_run_dir(Path::new("models/candidate.safetensors")),
            Path::new("models")
        );
    }

    #[test]
    fn fastchess_and_report_options_stay_identical() {
        let model = ModelEngine {
            name: "candidate",
            uci: Path::new("engine-zoo-uci"),
            run_dir: Path::new("run"),
            checkpoint: Path::new("model.safetensors"),
            simulations: 400,
            device: "cpu",
            opening_plies: Some(2),
        };
        assert_eq!(model.fastchess().options, model.spec().options);
    }
}
