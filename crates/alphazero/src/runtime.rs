//! Assembled AlphaZero engines for application-facing play and analysis.

use crate::experiment::InferenceConfig;
use crate::inference::{InferenceClient, InferenceService, InferenceSource};
use crate::representation::{ChessAzRepresentation, ChessAzState, ChessClassicRepresentation};
use crate::{ChessHistory, ChessRepetitionRules, ModelSpec, RepresentedEvaluator};
use anyhow::Result;
use chess::ChessMove;
use engine_core::agent::PolicyMode;
use games::{ChessGame, ChessRepetitionContext};
use search::{Mcts, PuctConfig, SearchConfig, SearchRequest, SearchResult};
use tch::Device;

/// A loaded chess engine. The service is owned only when this engine loaded
/// its model itself; registry users construct a client-backed instance instead.
pub struct ChessAlphaZeroEngine {
    _service: Option<InferenceService>,
    inner: ChessEngineKind,
}

enum ChessEngineKind {
    Classic(ChessEngine<ChessClassicRepresentation, ChessRepetitionRules>),
    H1(ChessEngine<ChessAzRepresentation<1>, ChessRepetitionRules>),
    H4(ChessEngine<ChessAzRepresentation<4>, ChessRepetitionRules>),
    H8(ChessEngine<ChessAzRepresentation<8>, ChessRepetitionRules>),
}

struct ChessEngine<R, Rules> {
    client: InferenceClient,
    marker: std::marker::PhantomData<fn() -> (R, Rules)>,
}

impl ChessAlphaZeroEngine {
    /// Loads a chess model and owns the corresponding inference service.
    pub fn open(
        model: ModelSpec,
        source: InferenceSource<'_>,
        device: Device,
        config: &InferenceConfig,
    ) -> Result<Self> {
        let service = InferenceService::load(&model, source, device, config)?;
        let mut engine = Self::from_client(model, service.client())?;
        engine._service = Some(service);

        Ok(engine)
    }

    /// Builds a short-lived search engine over a shared inference client.
    pub fn from_client(model: ModelSpec, client: InferenceClient) -> Result<Self> {
        let inner = match model.chess_history() {
            Some(ChessHistory::One) => ChessEngineKind::H1(ChessEngine::new(client)),
            Some(ChessHistory::Four) => ChessEngineKind::H4(ChessEngine::new(client)),
            Some(ChessHistory::Eight) => ChessEngineKind::H8(ChessEngine::new(client)),
            None if model.is_chess_classic() => ChessEngineKind::Classic(ChessEngine::new(client)),
            None => anyhow::bail!("model is not a chess model"),
        };

        Ok(Self {
            _service: None,
            inner,
        })
    }

    pub fn select_move(&mut self, game: &ChessGame, request: SearchRequest) -> Result<ChessMove> {
        match &mut self.inner {
            ChessEngineKind::Classic(engine) => engine.select_classic_move(game, request),
            ChessEngineKind::H1(engine) => engine.select_history_move(game, request),
            ChessEngineKind::H4(engine) => engine.select_history_move(game, request),
            ChessEngineKind::H8(engine) => engine.select_history_move(game, request),
        }
    }
}

impl<R, Rules> ChessEngine<R, Rules> {
    fn new(client: InferenceClient) -> Self {
        Self {
            client,
            marker: std::marker::PhantomData,
        }
    }
}

impl ChessEngine<ChessClassicRepresentation, ChessRepetitionRules> {
    fn select_classic_move(
        &mut self,
        game: &ChessGame,
        request: SearchRequest,
    ) -> Result<ChessMove> {
        let mut mcts = Mcts::new(
            RepresentedEvaluator::new(ChessClassicRepresentation, self.client.clone()),
            analysis_search(),
            ChessRepetitionRules,
        );
        let result = mcts.search(&game.position(), classic_context(game), request)?;

        Ok(select_result_move(&result, request.mode))
    }
}

impl<const HISTORY: usize> ChessEngine<ChessAzRepresentation<HISTORY>, ChessRepetitionRules> {
    fn select_history_move(
        &mut self,
        game: &ChessGame,
        request: SearchRequest,
    ) -> Result<ChessMove> {
        let state = ChessAzState::from_game(game);
        let mut mcts = Mcts::new(
            RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, self.client.clone()),
            analysis_search(),
            ChessRepetitionRules,
        );
        let result = mcts.search(&state, game.repetition_context(), request)?;

        Ok(select_result_move(&result, request.mode))
    }
}

fn analysis_search() -> SearchConfig {
    SearchConfig::Puct(PuctConfig::analysis_default(1))
}

fn classic_context(game: &ChessGame) -> ChessRepetitionContext<'_> {
    game.repetition_context()
}

fn select_result_move(result: &SearchResult<ChessMove>, mode: PolicyMode) -> ChessMove {
    if mode == PolicyMode::Explore {
        result.sample_move(&mut rand::rng())
    }
    else {
        result.best_move()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::notation::GameNotation;
    use games::{chess::ChessUciNotation, ChessRepetitionState};

    #[test]
    fn classic_engine_uses_authoritative_repetition_context() {
        let mut game = ChessGame::default();
        for text in ["b1c3", "b8c6", "c3b1", "c6b8"] {
            let mv = ChessUciNotation.parse_move(&game.position(), text).unwrap();
            game.play(mv);
        }

        let repetitions =
            classic_context(&game).occurrences_before_root(game.position().repetition_hash());
        assert_eq!(repetitions, 1);
    }
}
