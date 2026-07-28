use crate::representation::{ChessAzRepresentation, ChessAzState, ChessClassicRepresentation};
use crate::{AlphaZeroRepresentation, ChessRepetitionRules};
use engine_core::game::{GameState, TerminalValue};
use games::{ChessGame, ChessPosition};
use search::{NoExtraRules, SearchRules};
use std::marker::PhantomData;

/// Compile-time bridge between the shared self-play loop and native game rules.
pub trait SelfPlayDomain: Send + Sync + 'static {
    type Game;
    type State: GameState + Clone + Send + Sync + 'static;
    type Representation: AlphaZeroRepresentation<Self::State> + Default;
    type Rules: SearchRules<Self::State> + Default;

    fn initial_game() -> Self::Game;

    fn search_state(game: &Self::Game) -> Self::State;

    fn search_context<'a>(
        game: &'a Self::Game,
    ) -> <Self::Rules as SearchRules<Self::State>>::Context<'a>;

    fn play(game: &mut Self::Game, action: <Self::State as GameState>::Move);

    fn is_terminal(game: &Self::Game) -> bool;

    fn terminal_value(game: &Self::Game) -> Option<TerminalValue>;

    fn supports_resignation() -> bool {
        false
    }

    fn uses_evaluation_cache() -> bool {
        false
    }
}

/// Native-state domain for games without history-dependent search rules.
pub struct StandardSelfPlayDomain<G, Rep>(PhantomData<fn() -> (G, Rep)>);

impl<G, Rep> SelfPlayDomain for StandardSelfPlayDomain<G, Rep>
where
    G: GameState + Clone + Send + Sync + 'static,
    G::Move: Send + Sync,
    Rep: AlphaZeroRepresentation<G> + Default,
{
    type Game = G;
    type State = G;
    type Representation = Rep;
    type Rules = NoExtraRules;

    fn initial_game() -> Self::Game {
        G::initial()
    }

    fn search_state(game: &Self::Game) -> Self::State {
        game.clone()
    }

    fn search_context(_: &Self::Game) {}

    fn play(game: &mut Self::Game, action: G::Move) {
        game.play(action);
    }

    fn is_terminal(game: &Self::Game) -> bool {
        game.is_terminal()
    }

    fn terminal_value(game: &Self::Game) -> Option<TerminalValue> {
        game.terminal_value()
    }
}

/// Compatibility chess self-play with authoritative history and repetition rules.
pub struct ChessClassicSelfPlayDomain;

impl SelfPlayDomain for ChessClassicSelfPlayDomain {
    type Game = ChessGame;
    type State = ChessPosition;
    type Representation = ChessClassicRepresentation;
    type Rules = ChessRepetitionRules;

    fn initial_game() -> Self::Game {
        ChessGame::default()
    }

    fn search_state(game: &Self::Game) -> Self::State {
        game.position()
    }

    fn search_context<'a>(
        game: &'a Self::Game,
    ) -> <Self::Rules as SearchRules<Self::State>>::Context<'a> {
        game.repetition_context()
    }

    fn play(game: &mut Self::Game, action: chess::ChessMove) {
        game.play(action);
    }

    fn is_terminal(game: &Self::Game) -> bool {
        game.is_terminal()
    }

    fn terminal_value(game: &Self::Game) -> Option<TerminalValue> {
        game.terminal_value()
    }

    fn supports_resignation() -> bool {
        true
    }

    fn uses_evaluation_cache() -> bool {
        true
    }
}

/// Canonical chess self-play with fixed-size neural history states.
pub struct ChessCanonicalSelfPlayDomain<const HISTORY: usize>;

impl<const HISTORY: usize> SelfPlayDomain for ChessCanonicalSelfPlayDomain<HISTORY> {
    type Game = ChessGame;
    type State = ChessAzState<HISTORY>;
    type Representation = ChessAzRepresentation<HISTORY>;
    type Rules = ChessRepetitionRules;

    fn initial_game() -> Self::Game {
        ChessGame::default()
    }

    fn search_state(game: &Self::Game) -> Self::State {
        ChessAzState::from_game(game)
    }

    fn search_context<'a>(
        game: &'a Self::Game,
    ) -> <Self::Rules as SearchRules<Self::State>>::Context<'a> {
        game.repetition_context()
    }

    fn play(game: &mut Self::Game, action: chess::ChessMove) {
        game.play(action);
    }

    fn is_terminal(game: &Self::Game) -> bool {
        game.is_terminal()
    }

    fn terminal_value(game: &Self::Game) -> Option<TerminalValue> {
        game.terminal_value()
    }

    fn supports_resignation() -> bool {
        true
    }

    fn uses_evaluation_cache() -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::{ChessClassicSelfPlayDomain, SelfPlayDomain};
    use engine_core::notation::GameNotation;
    use games::{chess::ChessUciNotation, ChessGame, ChessRepetitionState};

    #[test]
    fn classic_chess_domain_keeps_authoritative_repetition_context() {
        let mut game = ChessGame::default();
        for notation in ["b1c3", "b8c6", "c3b1", "c6b8"] {
            let action = ChessUciNotation
                .parse_move(&game.position(), notation)
                .expect("test move is legal");
            game.play(action);
        }

        let state = ChessClassicSelfPlayDomain::search_state(&game);
        let context = ChessClassicSelfPlayDomain::search_context(&game);
        assert_eq!(state.repetition_hash(), game.position().repetition_hash());
        assert_eq!(context.occurrences_before_root(state.repetition_hash()), 1);
    }
}
