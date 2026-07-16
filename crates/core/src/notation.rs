use crate::game::GameState;

pub trait GameNotation<G: GameState> {
    fn parse_move(&self, state: &G, text: &str) -> Option<G::Move>;
    fn format_move(&self, state: &G, mv: G::Move) -> String;
}
