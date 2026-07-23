//! Build and run Fastchess matches without embedding tournament policy in a CLI.

use anyhow::{bail, Context, Result};
use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Engine {
    pub command: PathBuf,
    pub name: String,
    pub directory: Option<PathBuf>,
    pub options: Vec<(String, String)>,
    pub nodes: Option<usize>,
}

impl Engine {
    pub fn new(command: impl Into<PathBuf>, name: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            name: name.into(),
            directory: None,
            options: Vec::new(),
            nodes: None,
        }
    }

    pub fn directory(mut self, directory: impl Into<PathBuf>) -> Self {
        self.directory = Some(directory.into());
        self
    }

    pub fn option(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.options.push((name.into(), value.into()));
        self
    }

    pub fn nodes(mut self, nodes: usize) -> Self {
        self.nodes = Some(nodes);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum OpeningFormat {
    Pgn,
    Epd,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Openings {
    pub file: PathBuf,
    pub format: OpeningFormat,
    pub plies: Option<u32>,
}

impl Openings {
    pub fn pgn(file: impl Into<PathBuf>) -> Self {
        Self {
            file: file.into(),
            format: OpeningFormat::Pgn,
            plies: None,
        }
    }

    pub fn epd(file: impl Into<PathBuf>) -> Self {
        Self {
            file: file.into(),
            format: OpeningFormat::Epd,
            plies: None,
        }
    }

    pub fn plies(mut self, plies: u32) -> Self {
        self.plies = Some(plies);
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastchessCommand {
    pub executable: PathBuf,
    pub engines: [Engine; 2],
    pub time_control: String,
    pub rounds: u32,
    pub concurrency: Option<u32>,
    pub openings: Option<Openings>,
    pub pgn_output: Option<PathBuf>,
    pub extra_args: Vec<OsString>,
}

impl FastchessCommand {
    pub fn new(executable: impl Into<PathBuf>, white: Engine, black: Engine) -> Self {
        Self {
            executable: executable.into(),
            engines: [white, black],
            time_control: "10+0.1".into(),
            rounds: 1,
            concurrency: None,
            openings: None,
            pgn_output: None,
            extra_args: Vec::new(),
        }
    }

    pub fn time_control(mut self, tc: impl Into<String>) -> Self {
        self.time_control = tc.into();
        self
    }
    pub fn rounds(mut self, rounds: u32) -> Self {
        self.rounds = rounds;
        self
    }
    pub fn concurrency(mut self, concurrency: u32) -> Self {
        self.concurrency = Some(concurrency);
        self
    }
    pub fn openings(mut self, openings: Openings) -> Self {
        self.openings = Some(openings);
        self
    }
    pub fn pgn_output(mut self, path: impl Into<PathBuf>) -> Self {
        self.pgn_output = Some(path.into());
        self
    }
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.extra_args.push(arg.into());
        self
    }

    /// The exact argv passed to Fastchess. `-repeat` is deliberate: it pairs
    /// every opening with a colour reversal.
    pub fn args(&self) -> Result<Vec<OsString>> {
        if self.rounds == 0 {
            bail!("Fastchess rounds must be greater than zero")
        }
        if self.concurrency == Some(0) {
            bail!("Fastchess concurrency must be greater than zero")
        }
        if self.time_control.trim().is_empty() {
            bail!("Fastchess time control must not be empty")
        }
        if self.engines[0].name == self.engines[1].name {
            bail!("Fastchess engine names must differ")
        }
        if self.engines.iter().any(|engine| engine.nodes == Some(0)) {
            bail!("Fastchess engine nodes must be greater than zero")
        }
        let mut args = Vec::new();
        for engine in &self.engines {
            args.extend([
                "-engine".into(),
                format!("cmd={}", engine.command.display()).into(),
                format!("name={}", engine.name).into(),
            ]);
            if let Some(dir) = &engine.directory {
                args.push(format!("dir={}", dir.display()).into());
            }
            for (name, value) in &engine.options {
                args.push(format!("option.{name}={value}").into());
            }
            if let Some(nodes) = engine.nodes {
                args.push(format!("nodes={nodes}").into());
            }
        }
        args.extend([
            "-each".into(),
            format!("tc={}", self.time_control).into(),
            "-rounds".into(),
            self.rounds.to_string().into(),
            "-repeat".into(),
        ]);
        if let Some(concurrency) = self.concurrency {
            args.extend(["-concurrency".into(), concurrency.to_string().into()]);
        }
        if let Some(openings) = &self.openings {
            let format = match openings.format {
                OpeningFormat::Pgn => "pgn",
                OpeningFormat::Epd => "epd",
            };
            args.extend([
                "-openings".into(),
                format!("file={}", openings.file.display()).into(),
                format!("format={format}").into(),
                "order=sequential".into(),
            ]);
            if let Some(plies) = openings.plies {
                args.push(format!("plies={plies}").into());
            }
        }
        if let Some(pgn) = &self.pgn_output {
            args.extend([
                "-pgnout".into(),
                format!("file={}", pgn.display()).into(),
                "notation=san".into(),
                "append=false".into(),
            ]);
        }
        args.extend(self.extra_args.iter().cloned());
        Ok(args)
    }

    pub fn command_line(&self) -> Result<String> {
        let mut parts = vec![shell_quote(&self.executable.display().to_string())];
        parts.extend(
            self.args()?
                .into_iter()
                .map(|a| shell_quote(&a.to_string_lossy())),
        );
        Ok(parts.join(" "))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FastchessRun {
    pub args: Vec<OsString>,
    pub command_line: String,
    pub status: i32,
    pub stdout: String,
    pub stderr: String,
    pub pgn: Option<String>,
    pub version: Option<String>,
}

pub fn run(command: &FastchessCommand) -> Result<FastchessRun> {
    let args = command.args()?;
    if let Some(pgn) = &command.pgn_output {
        if let Some(parent) = pgn.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent).with_context(|| {
                format!("creating Fastchess PGN directory {}", parent.display())
            })?;
        }
        // A failed or misconfigured run must not be able to report the PGN
        // produced by an earlier match at the same path.
        if pgn.exists() {
            fs::remove_file(pgn)
                .with_context(|| format!("removing stale Fastchess PGN {}", pgn.display()))?;
        }
    }
    let version = version(&command.executable).ok();
    let mut child = Command::new(&command.executable)
        .args(&args)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("starting Fastchess at {}", command.executable.display()))?;
    let stdout = child.stdout.take().expect("piped Fastchess stdout");
    let stderr = child.stderr.take().expect("piped Fastchess stderr");
    let stdout_thread = std::thread::spawn(move || {
        let mut captured = String::new();
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if !is_pinning_warning(&line) {
                println!("{line}");
            }
            captured.push_str(&line);
            captured.push('\n');
        }
        captured
    });
    let stderr_thread = std::thread::spawn(move || {
        let mut captured = String::new();
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if !is_pinning_warning(&line) {
                eprintln!("fastchess: {line}");
            }
            captured.push_str(&line);
            captured.push('\n');
        }
        captured
    });
    let status = child.wait()?;
    let output = Output {
        status,
        stdout: stdout_thread.join().unwrap_or_default().into_bytes(),
        stderr: stderr_thread.join().unwrap_or_default().into_bytes(),
    };
    finish_run(command, args, output, version)
}

fn is_pinning_warning(line: &str) -> bool {
    line.contains("pin()")
        || line.contains("pin memory")
        || line.contains("pin_memory()")
        || line.contains("is_pinned()")
}

fn finish_run(
    command: &FastchessCommand,
    args: Vec<OsString>,
    output: Output,
    version: Option<String>,
) -> Result<FastchessRun> {
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let pgn = command
        .pgn_output
        .as_ref()
        .filter(|path| path.is_file())
        .map(fs::read_to_string)
        .transpose()
        .context("reading Fastchess PGN output")?;
    Ok(FastchessRun {
        args,
        command_line: command.command_line()?,
        status: output.status.code().unwrap_or(-1),
        stdout,
        stderr,
        pgn,
        version,
    })
}

pub fn version(executable: impl AsRef<Path>) -> Result<String> {
    let output = Command::new(executable.as_ref())
        .arg("-version")
        .output()
        .with_context(|| format!("checking Fastchess at {}", executable.as_ref().display()))?;
    if !output.status.success() {
        bail!("Fastchess version command exited with {}", output.status)
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let text = if stdout.trim().is_empty() {
        stderr
    }
    else {
        stdout
    };
    Ok(text.trim().to_owned())
}

mod result;

pub use result::{parse_score, parse_wdl, Wdl};

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "_./:=+-".contains(c))
    {
        value.to_owned()
    }
    else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

#[cfg(test)]
#[path = "fastchess_tests.rs"]
mod tests;
