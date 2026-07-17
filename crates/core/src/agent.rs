use crate::game::GameState;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolicyMode {
    Deterministic,
    Explore,
}

/// Anything that can pick a move
pub trait Agent<G: GameState> {
    fn select_move(&mut self, game: &G, mode: PolicyMode) -> G::Move;

    fn select_deterministic_move(&mut self, game: &G) -> G::Move {
        self.select_move(game, PolicyMode::Deterministic)
    }
}
