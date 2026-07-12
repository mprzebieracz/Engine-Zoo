use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CreateJob {
    pub candidate: String,
    pub suite: Suite,
    #[serde(default)]
    pub baseline: Option<String>,
    pub simulations: usize,
    #[serde(default = "default_rounds")]
    pub rounds: u32,
    #[serde(default = "default_nodes")]
    pub stockfish_nodes: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Suite {
    Puzzle,
    Stockfish,
    Arena,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobStatus {
    Queued,
    Running,
    Succeeded,
    Failed,
    Interrupted,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub id: String,
    pub candidate: String,
    pub suite: Suite,
    pub baseline: Option<String>,
    pub simulations: usize,
    pub rounds: u32,
    pub stockfish_nodes: usize,
    pub status: JobStatus,
    pub created_at: u64,
    pub updated_at: u64,
    pub error: Option<String>,
    #[serde(default)]
    pub artifacts: Vec<String>,
    pub result: Option<Value>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Catalog {
    pub checkpoints: Vec<String>,
    pub suites: Vec<SuiteInfo>,
    pub preflight: Preflight,
}

#[derive(Clone, Debug, Serialize)]
pub struct SuiteInfo {
    pub id: Suite,
    pub label: &'static str,
    pub defaults: Value,
}

#[derive(Clone, Debug, Serialize)]
pub struct Preflight {
    pub puzzle: ToolState,
    pub stockfish: ToolState,
    pub arena: ToolState,
}

#[derive(Clone, Debug, Serialize)]
pub struct ToolState {
    pub available: bool,
    pub message: String,
}

fn default_rounds() -> u32 {
    2
}

fn default_nodes() -> usize {
    3000
}
