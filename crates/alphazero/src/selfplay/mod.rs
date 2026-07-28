//! Deterministic AlphaZero self-play orchestration.

mod config;
mod coordinator;
mod domain;
mod worker;

pub use config::{
    GumbelMoveSelection, ResignationConfig, SearchBudget, SearchBudgetSchedule, SelfPlayConfig,
    TemperaturePhase, TemperatureSchedule,
};
pub use coordinator::{
    CompletedGame, GameRequest, SelfPlayCoordinator, SelfPlayEpoch, SelfPlayStats, SelfPlayWorker,
    SelfPlayWorkerFactory,
};
pub use domain::{
    ChessCanonicalSelfPlayDomain, ChessClassicSelfPlayDomain, SelfPlayDomain,
    StandardSelfPlayDomain,
};
pub use worker::DomainSelfPlayWorkerFactory;

use rand::distr::Distribution;
use rand::Rng;
use search::SearchResult;

/// Samples `p^(1/T)` without giving mass to zero-probability actions.
/// `None` selects the policy argmax.
pub fn select_temperature_action<M: Copy, R: Rng + ?Sized>(
    result: &SearchResult<M>,
    temperature: Option<f32>,
    rng: &mut R,
) -> M {
    let Some(temperature) = temperature
    else {
        return result.best_move();
    };
    debug_assert!(temperature.is_finite() && temperature > 0.0);
    if temperature == 1.0 {
        return result.sample_move(rng);
    }

    let exponent = 1.0_f64 / f64::from(temperature);
    let weights: Vec<f64> = result
        .policy
        .iter()
        .map(|&(_, probability)| f64::from(probability).powf(exponent))
        .collect();
    let index = rand::distr::weighted::WeightedIndex::new(weights)
        .expect("search of a non-terminal position returns a non-empty policy")
        .sample(rng);
    result.policy[index].0
}

#[cfg(test)]
mod tests;
