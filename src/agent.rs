use crate::game::{Action, Game};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyMode {
    Deterministic,
    Explore,
}

/// Anything that can pick a move
pub trait Agent<G: Game> {
    fn act_with_mode(&mut self, game: &G, mode: PolicyMode) -> Action;

    fn act(&mut self, game: &G) -> Action {
        self.act_with_mode(game, PolicyMode::Deterministic)
    }
}
