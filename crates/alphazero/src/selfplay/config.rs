use anyhow::{ensure, Result};
use engine_core::agent::PolicyMode;
use search::{SearchConfig, SearchRequest};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchBudget {
    Puct {
        simulations: usize,
    },
    Gumbel {
        simulations: usize,
        max_considered_actions: usize,
    },
}

impl SearchBudget {
    pub const fn simulations(self) -> usize {
        match self {
            Self::Puct { simulations, .. } | Self::Gumbel { simulations, .. } => simulations,
        }
    }

    fn matches(self, search: &SearchConfig) -> bool {
        matches!(
            (self, search),
            (Self::Puct { .. }, SearchConfig::Puct(_))
                | (Self::Gumbel { .. }, SearchConfig::RootGumbelPuct(_))
                | (Self::Gumbel { .. }, SearchConfig::FullGumbel(_))
        )
    }

    fn validate(self) -> Result<()> {
        match self {
            Self::Puct { simulations } => {
                ensure!(simulations > 0, "PUCT simulations must be positive");
            }
            Self::Gumbel {
                simulations,
                max_considered_actions,
            } => {
                ensure!(simulations > 0, "Gumbel simulations must be positive");
                ensure!(
                    max_considered_actions > 0,
                    "Gumbel max_considered_actions must be positive"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum SearchBudgetSchedule {
    Fixed(SearchBudget),
    PlayoutCapRandomization {
        full: SearchBudget,
        fast: SearchBudget,
        full_probability: f32,
        fast_policy_weight: f32,
    },
}

impl SearchBudgetSchedule {
    pub fn fixed(search: &SearchConfig) -> Self {
        match search {
            SearchConfig::Puct(_) => Self::Fixed(SearchBudget::Puct { simulations: 800 }),
            SearchConfig::RootGumbelPuct(_) | SearchConfig::FullGumbel(_) => {
                Self::Fixed(SearchBudget::Gumbel {
                    simulations: 128,
                    max_considered_actions: 16,
                })
            }
        }
    }

    pub fn choose<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> (SearchBudget, f32, bool) {
        match *self {
            Self::Fixed(budget) => (budget, 1.0, true),
            Self::PlayoutCapRandomization {
                full,
                full_probability,
                ..
            } if rng.random_bool(f64::from(full_probability)) => (full, 1.0, true),
            Self::PlayoutCapRandomization {
                fast,
                fast_policy_weight,
                ..
            } => (fast, fast_policy_weight, false),
        }
    }

    fn validate(&self, search: &SearchConfig) -> Result<()> {
        match *self {
            Self::Fixed(budget) => {
                budget.validate()?;
                ensure!(
                    budget.matches(search),
                    "search budget must use the configured search algorithm"
                );
            }
            Self::PlayoutCapRandomization {
                full,
                fast,
                full_probability,
                fast_policy_weight,
            } => {
                full.validate()?;
                fast.validate()?;
                ensure!(
                    full.matches(search) && fast.matches(search),
                    "full and fast budgets must use the configured search algorithm"
                );
                ensure!(
                    full_probability.is_finite() && (0.0..=1.0).contains(&full_probability),
                    "full_probability must be finite and in [0, 1]"
                );
                ensure!(
                    fast_policy_weight.is_finite() && fast_policy_weight >= 0.0,
                    "fast_policy_weight must be finite and non-negative"
                );
            }
        }
        Ok(())
    }
}

impl From<SearchBudget> for search::SearchBudget {
    fn from(budget: SearchBudget) -> Self {
        match budget {
            SearchBudget::Puct { simulations } => Self::Puct { simulations },
            SearchBudget::Gumbel {
                simulations,
                max_considered_actions,
            } => Self::Gumbel {
                simulations,
                max_considered_actions,
            },
        }
    }
}

pub(crate) fn search_request(budget: SearchBudget, mode: PolicyMode) -> SearchRequest {
    SearchRequest {
        mode,
        budget: budget.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemperaturePhase {
    pub until_ply_exclusive: usize,
    pub temperature: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TemperatureSchedule {
    pub phases: Vec<TemperaturePhase>,
}

impl TemperatureSchedule {
    pub fn at_ply(&self, ply: usize) -> Option<f32> {
        self.phases
            .iter()
            .find(|phase| ply < phase.until_ply_exclusive)
            .map(|phase| phase.temperature)
    }

    fn validate(&self) -> Result<()> {
        let mut previous = 0;
        for phase in &self.phases {
            ensure!(
                phase.until_ply_exclusive > previous,
                "temperature phase boundaries must increase"
            );
            ensure!(
                phase.temperature.is_finite() && phase.temperature > 0.0,
                "temperature must be finite and positive"
            );
            previous = phase.until_ply_exclusive;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GumbelMoveSelection {
    #[default]
    ProposedAction,
    ImprovedPolicyTemperature,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResignationConfig {
    pub enabled: bool,
    pub threshold: f32,
    pub consecutive_moves: usize,
    pub minimum_ply: usize,
    pub disable_probability: f32,
}

impl Default for ResignationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: -0.95,
            consecutive_moves: 3,
            minimum_ply: 60,
            disable_probability: 0.0,
        }
    }
}

impl ResignationConfig {
    fn validate(self) -> Result<()> {
        ensure!(
            self.threshold.is_finite(),
            "resignation threshold must be finite"
        );
        ensure!(
            self.consecutive_moves > 0,
            "resignation consecutive_moves must be positive"
        );
        ensure!(
            self.disable_probability.is_finite() && (0.0..=1.0).contains(&self.disable_probability),
            "resignation disable_probability must be in [0, 1]"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SelfPlayConfig {
    pub num_games: usize,
    pub threads: usize,
    pub max_moves: usize,
    pub progress_every: usize,
    pub search: SearchConfig,
    pub budget_schedule: SearchBudgetSchedule,
    pub temperature: TemperatureSchedule,
    pub gumbel_move_selection: GumbelMoveSelection,
    pub resignation: ResignationConfig,
    pub evaluation_cache_entries: usize,
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        let search = SearchConfig::default();
        Self {
            num_games: 100,
            threads: std::thread::available_parallelism().map_or(1, |threads| threads.get()),
            max_moves: 256,
            progress_every: 25,
            budget_schedule: SearchBudgetSchedule::fixed(&search),
            search,
            temperature: TemperatureSchedule {
                phases: vec![TemperaturePhase {
                    until_ply_exclusive: 30,
                    temperature: 1.0,
                }],
            },
            gumbel_move_selection: GumbelMoveSelection::ProposedAction,
            resignation: ResignationConfig::default(),
            evaluation_cache_entries: 0,
        }
    }
}

impl SelfPlayConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(self.num_games > 0, "self-play num_games must be positive");
        ensure!(self.threads > 0, "self-play threads must be positive");
        ensure!(self.max_moves > 0, "self-play max_moves must be positive");
        self.search.validate().map_err(anyhow::Error::msg)?;
        self.budget_schedule.validate(&self.search)?;
        self.temperature.validate()?;
        self.resignation.validate()
    }
}
