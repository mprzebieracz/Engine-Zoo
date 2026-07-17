use engine_core::game::{GameState, TerminalValue};
use games::{ChessRepetitionContext, ChessRepetitionState};

/// Result of applying rules external to the game state's local terminal test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuleResult {
    Continue,
    Terminal(TerminalValue),
}

/// Adjudication and per-search state used while traversing an MCTS tree.
pub trait SearchRules<G: GameState>: Send + 'static {
    type Context<'a>: Copy
    where
        Self: 'a,
        G: 'a;

    type PathState: Default;
    type NodeMeta: Default;

    /// Reuses the path allocation and initializes it for one traversal.
    fn reset_path<'a>(&self, context: Self::Context<'a>, root: &G, path: &mut Self::PathState);

    /// Called after the state corresponding to a tree node has been reached.
    fn enter_state<'a>(
        &self,
        context: Self::Context<'a>,
        state: &mut G,
        path: &mut Self::PathState,
        meta: &mut Self::NodeMeta,
    ) -> RuleResult;
}

/// Standard search has no history-dependent adjudication or metadata.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoExtraRules;

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
        }
        else {
            state.repetition_hash()
        };
        let repetitions_before = if meta.initialized {
            meta.repetitions_before_current
        }
        else {
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

impl<G: GameState> SearchRules<G> for NoExtraRules {
    type Context<'a>
        = ()
    where
        G: 'a;
    type PathState = ();
    type NodeMeta = ();

    fn reset_path<'a>(&self, _: Self::Context<'a>, _: &G, _: &mut Self::PathState) {}

    fn enter_state<'a>(
        &self,
        _: Self::Context<'a>,
        _: &mut G,
        _: &mut Self::PathState,
        _: &mut Self::NodeMeta,
    ) -> RuleResult {
        RuleResult::Continue
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess::ChessMove;
    use engine_core::game::{Game, TerminalValue};
    use games::ChessGame;

    #[derive(Clone, Copy, Debug, Default)]
    struct State;

    #[derive(Clone, Copy, Debug, Default)]
    struct RepetitionState {
        hash: u64,
        reversible_plies: usize,
        repetitions_before: u8,
    }

    impl GameState for State {
        type Move = u8;

        fn initial() -> Self {
            Self
        }
        fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
            [].into_iter()
        }
        fn play(&mut self, _: Self::Move) {}
        fn terminal_value(&self) -> Option<TerminalValue> {
            None
        }
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
            let action = game.parse_move(mv).unwrap();
            game.step(action);
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
    fn no_extra_rules_have_no_state_and_continue() {
        let rules = NoExtraRules;
        let mut path = ();
        let mut meta = ();
        let mut state = State;
        rules.reset_path((), &state, &mut path);
        assert_eq!(
            rules.enter_state((), &mut state, &mut path, &mut meta),
            RuleResult::Continue
        );
        assert_eq!(std::mem::size_of_val(&meta), 0);
    }

    #[test]
    fn chess_context_contributes_root_and_non_root_occurrences() {
        let mut game = ChessGame::default();
        let non_root_hash = {
            let action = game.parse_move("b1c3").unwrap();
            game.step(action);
            game.position().hash()
        };
        for mv in ["b8c6", "c3b1", "c6b8"] {
            let action = game.parse_move(mv).unwrap();
            game.step(action);
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
        let mut state = game.history_state::<4>();

        assert_eq!(
            ChessRepetitionRules.enter_state(
                context,
                &mut state,
                &mut Vec::new(),
                &mut ChessNodeMeta::default(),
            ),
            RuleResult::Continue
        );
        let mut current_repetitions = None;
        state.for_each_frame(|index, _, repetitions| {
            if index == 0 {
                current_repetitions = Some(repetitions);
            }
        });
        assert_eq!(current_repetitions, Some(1));
    }
}
