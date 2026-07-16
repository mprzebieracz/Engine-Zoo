use engine_core::game::{GameState, TerminalValue};

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
    use engine_core::game::TerminalValue;

    #[derive(Clone, Copy, Debug, Default)]
    struct State;

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
}
