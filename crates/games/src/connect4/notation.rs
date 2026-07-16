use super::{Connect4, Connect4Move};
use engine_core::{notation::GameNotation, GameState};

#[derive(Clone, Copy, Debug, Default)]
pub struct Connect4Notation;

impl GameNotation<Connect4> for Connect4Notation {
    fn parse_move(&self, state: &Connect4, text: &str) -> Option<Connect4Move> {
        let column = text.trim().parse().ok()?;
        let mv = Connect4Move::new(column)?;
        state.legal_moves().any(|legal| legal == mv).then_some(mv)
    }

    fn format_move(&self, _state: &Connect4, mv: Connect4Move) -> String {
        mv.column().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notation_parses_and_formats_legal_moves() {
        let game = Connect4::default();
        let mv = Connect4Notation.parse_move(&game, " 3 ").unwrap();
        assert_eq!(Connect4Notation.format_move(&game, mv), "3");
        assert_eq!(Connect4Notation.parse_move(&game, "7"), None);
    }
}
