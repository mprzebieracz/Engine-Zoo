use engine_core::game::TerminalValue;
use games::{ChessRepetitionContext, ChessRepetitionState};
use search::{RuleResult, SearchRules};

#[derive(Clone, Copy, Debug, Default)]
pub struct ChessRepetitionRules;

#[derive(Clone, Copy, Debug, Default)]
pub struct ChessNodeMeta {
    hash: u64,
    repetitions_before_current: u8,
    initialized: bool,
}

impl<G: ChessRepetitionState> SearchRules<G> for ChessRepetitionRules {
    type Context<'a>
        = ChessRepetitionContext<'a>
    where
        G: 'a;
    type PathState = Vec<u64>;
    type NodeMeta = ChessNodeMeta;

    fn reset_path<'a>(&self, _: Self::Context<'a>, _: &G, path: &mut Self::PathState) {
        path.clear();
        path.reserve(256);
    }

    fn enter_state<'a>(
        &self,
        context: Self::Context<'a>,
        state: &mut G,
        path: &mut Self::PathState,
        meta: &mut Self::NodeMeta,
    ) -> RuleResult {
        let hash = if meta.initialized {
            meta.hash
        } else {
            state.repetition_hash()
        };
        let repetitions_before = if meta.initialized {
            meta.repetitions_before_current
        } else {
            let mut count = context.occurrences_before_root(hash);
            for seen in path.iter().rev().take(state.reversible_plies()) {
                count = count.saturating_add(u8::from(*seen == hash));
            }
            meta.hash = hash;
            meta.repetitions_before_current = count;
            meta.initialized = true;
            count
        };
        state.set_current_repetitions_before(repetitions_before);
        let result = (repetitions_before >= 2).then_some(RuleResult::Terminal(TerminalValue::Draw));
        path.push(hash);
        result.unwrap_or(RuleResult::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::ChessMove;
    use engine_core::{game::TerminalValue, notation::GameNotation, GameState};
    use games::{chess::ChessUciNotation, ChessGame};

    #[derive(Clone, Copy, Debug, Default)]
    struct RepetitionState {
        hash: u64,
        reversible_plies: usize,
        repetitions_before: u8,
    }

    impl GameState for RepetitionState {
        type Move = ChessMove;

        fn initial() -> Self {
            Self::default()
        }
        fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
            std::iter::empty()
        }
        fn play(&mut self, _: Self::Move) {}
        fn terminal_value(&self) -> Option<TerminalValue> {
            None
        }
    }

    impl ChessRepetitionState for RepetitionState {
        fn repetition_hash(&self) -> u64 {
            self.hash
        }

        fn reversible_plies(&self) -> usize {
            self.reversible_plies
        }

        fn set_current_repetitions_before(&mut self, count: u8) {
            self.repetitions_before = count;
        }
    }

    fn play_cycle(game: &mut ChessGame) {
        for mv in ["b1c3", "b8c6", "c3b1", "c6b8"] {
            let mv = ChessUciNotation
                .parse_move(&game.position_state(), mv)
                .unwrap();
            game.play(mv);
        }
    }

    fn enter(
        context: ChessRepetitionContext<'_>,
        state: &mut RepetitionState,
        path: &mut Vec<u64>,
        meta: &mut ChessNodeMeta,
    ) -> RuleResult {
        ChessRepetitionRules.enter_state(context, state, path, meta)
    }

    #[test]
    fn chess_context_contributes_root_and_non_root_occurrences() {
        let mut game = ChessGame::default();
        let non_root_hash = {
            let mv = ChessUciNotation
                .parse_move(&game.position_state(), "b1c3")
                .unwrap();
            game.play(mv);
            game.position().hash()
        };
        for mv in ["b8c6", "c3b1", "c6b8"] {
            let mv = ChessUciNotation
                .parse_move(&game.position_state(), mv)
                .unwrap();
            game.play(mv);
        }
        let context = game.repetition_context();
        let root_hash = game.position().hash();

        for hash in [root_hash, non_root_hash] {
            let mut state = RepetitionState {
                hash,
                ..Default::default()
            };
            assert_eq!(
                enter(
                    context,
                    &mut state,
                    &mut vec![],
                    &mut ChessNodeMeta::default()
                ),
                RuleResult::Continue
            );
            assert_eq!(state.repetitions_before, 1);
        }
    }

    #[test]
    fn root_to_descendant_cycle_reaches_threefold_draw() {
        let game = ChessGame::default();
        let context = game.repetition_context();
        let mut path = Vec::new();

        for hash in [10, 20, 10, 20] {
            let mut state = RepetitionState {
                hash,
                reversible_plies: path.len(),
                ..Default::default()
            };
            assert_eq!(
                enter(
                    context,
                    &mut state,
                    &mut path,
                    &mut ChessNodeMeta::default()
                ),
                RuleResult::Continue
            );
        }
        let mut third = RepetitionState {
            hash: 10,
            reversible_plies: path.len(),
            ..Default::default()
        };
        assert_eq!(
            enter(
                context,
                &mut third,
                &mut path,
                &mut ChessNodeMeta::default()
            ),
            RuleResult::Terminal(TerminalValue::Draw)
        );
        assert_eq!(third.repetitions_before, 2);
        assert_eq!(path, [10, 20, 10, 20, 10]);
    }

    #[test]
    fn reversible_plies_limit_path_suffix_scanning() {
        let game = ChessGame::default();
        let mut state = RepetitionState {
            hash: 7,
            reversible_plies: 2,
            ..Default::default()
        };
        let mut path = vec![7, 8, 7];

        assert_eq!(
            enter(
                game.repetition_context(),
                &mut state,
                &mut path,
                &mut ChessNodeMeta::default(),
            ),
            RuleResult::Continue
        );
        assert_eq!(state.repetitions_before, 1);
    }

    #[test]
    fn cached_meta_revisit_updates_feature_and_pushes_cached_hash() {
        let game = ChessGame::default();
        let context = game.repetition_context();
        let mut meta = ChessNodeMeta::default();
        let mut first = RepetitionState {
            hash: 42,
            reversible_plies: 1,
            ..Default::default()
        };
        enter(context, &mut first, &mut vec![42], &mut meta);
        assert_eq!(first.repetitions_before, 1);

        let mut revisit = RepetitionState {
            hash: 99,
            repetitions_before: 255,
            ..Default::default()
        };
        let mut revisit_path = vec![1];
        assert_eq!(
            enter(context, &mut revisit, &mut revisit_path, &mut meta),
            RuleResult::Continue
        );
        assert_eq!(revisit.repetitions_before, 1);
        assert_eq!(revisit_path, [1, 42]);
    }

    #[test]
    fn reset_path_retains_reserved_allocation() {
        let game = ChessGame::default();
        let context = game.repetition_context();
        let mut path = Vec::new();
        ChessRepetitionRules.reset_path(context, &RepetitionState::default(), &mut path);
        path.extend(0..128);
        let pointer = path.as_ptr();
        let capacity = path.capacity();

        ChessRepetitionRules.reset_path(context, &RepetitionState::default(), &mut path);
        assert!(path.is_empty());
        assert_eq!(path.as_ptr(), pointer);
        assert_eq!(path.capacity(), capacity);
        assert!(capacity >= 256);
    }

    #[test]
    fn history_state_enter_writes_current_repetition_feature() {
        let mut game = ChessGame::default();
        play_cycle(&mut game);
        let context = game.repetition_context();
        let mut state = game.position();

        assert_eq!(
            ChessRepetitionRules.enter_state(
                context,
                &mut state,
                &mut Vec::new(),
                &mut ChessNodeMeta::default(),
            ),
            RuleResult::Continue
        );
    }
}
