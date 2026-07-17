use super::{Action, AlphaZeroRepresentation};
use engine_core::GameState;
use games::{Connect4, Connect4Move};

#[derive(Clone, Copy, Debug, Default)]
pub struct Connect4AzRepresentation;

impl AlphaZeroRepresentation<Connect4> for Connect4AzRepresentation {
    const STATE_SHAPE: [usize; 3] = [1, 6, 7];
    const ACTION_SIZE: usize = 7;

    fn encode_state(&self, state: &Connect4, output: &mut [f32]) {
        assert_eq!(
            output.len(),
            Self::state_size(),
            "invalid state buffer length"
        );
        let own = state.position_bits();
        let opponent = own ^ state.occupied_bits();
        for row_top in 0..6 {
            for col in 0..7 {
                let bit = 1u64 << (col * 7 + (5 - row_top));
                output[row_top * 7 + col] = if own & bit != 0 {
                    1.0
                } else if opponent & bit != 0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
    }

    fn move_to_action(&self, _state: &Connect4, mv: Connect4Move) -> Action {
        Action::new(mv.column() as u32)
    }

    fn action_to_move(&self, state: &Connect4, action: Action) -> Option<Connect4Move> {
        let mv = Connect4Move::new(u8::try_from(action.index()).ok()?)?;
        state.legal_moves().any(|legal| legal == mv).then_some(mv)
    }

    fn encoded_state_key(&self, state: &Connect4) -> u64 {
        let mut key = state.position_bits().wrapping_mul(0x9e37_79b9_7f4a_7c15);
        key ^= state.occupied_bits().rotate_left(23);
        key ^ u64::from(state.is_ongoing())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::GameState;

    #[test]
    fn golden_encoding_and_legal_roundtrips() {
        let r = Connect4AzRepresentation;
        let mut game = Connect4::initial();
        let mut seed = 0x1234_5678u32;
        for _ in 0..30 {
            let moves: Vec<_> = game.legal_moves().collect();
            let mut actions = std::collections::HashSet::new();
            for mv in moves.iter().copied() {
                let action = r.move_to_action(&game, mv);
                assert!(actions.insert(action));
                assert_eq!(r.action_to_move(&game, action), Some(mv));
            }
            if moves.is_empty() {
                break;
            }
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            game.play(moves[seed as usize % moves.len()]);
        }
    }

    #[test]
    fn invalid_illegal_actions_and_keys_are_handled() {
        let r = Connect4AzRepresentation;
        let game = Connect4::initial();
        assert!(r.action_to_move(&game, Action::new(7)).is_none());
        let full = Connect4::from_moves(
            &[0, 1, 0, 1, 0, 1, 0]
                .into_iter()
                .map(|column| Connect4Move::new(column).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(r.action_to_move(&full, Action::new(0)).is_none());
        let before = r.encoded_state_key(&game);
        let mut next = game;
        next.play(Connect4Move::new(0).unwrap());
        assert_ne!(before, r.encoded_state_key(&next));
    }
}
