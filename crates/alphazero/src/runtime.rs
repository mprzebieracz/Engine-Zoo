//! Assembled AlphaZero engines for application-facing play and analysis.

use crate::experiment::InferenceConfig;
use crate::inference::{InferenceClient, InferenceService, InferenceSource};
use crate::representation::{ChessAzRepresentation, ChessAzState, ChessClassicRepresentation};
use crate::{ChessHistory, ChessRepetitionRules, ModelSpec, RepresentedEvaluator};
use anyhow::Result;
use chess::ChessMove;
use engine_core::agent::PolicyMode;
use games::{ChessGame, ChessPosition, ChessRepetitionContext};
use search::{Mcts, SearchAlgorithm, SearchConfig, SearchRequest, SearchResult};
use tch::Device;

type ClassicMcts = Mcts<
    ChessPosition,
    RepresentedEvaluator<ChessPosition, ChessClassicRepresentation, InferenceClient>,
    ChessRepetitionRules,
>;
type HistoryMcts<const HISTORY: usize> = Mcts<
    ChessAzState<HISTORY>,
    RepresentedEvaluator<ChessAzState<HISTORY>, ChessAzRepresentation<HISTORY>, InferenceClient>,
    ChessRepetitionRules,
>;

/// A loaded chess engine. The service is owned only when this engine loaded
/// its model itself; registry users construct a client-backed instance instead.
pub struct ChessAlphaZeroEngine {
    _service: Option<InferenceService>,
    inner: ChessEngineKind,
}

enum ChessEngineKind {
    Classic(ClassicMcts),
    H1(HistoryMcts<1>),
    H4(HistoryMcts<4>),
    H8(HistoryMcts<8>),
}

impl ChessAlphaZeroEngine {
    /// Loads a chess model and owns the corresponding inference service.
    pub fn open(
        model: ModelSpec,
        source: InferenceSource<'_>,
        device: Device,
        config: &InferenceConfig,
        search: SearchConfig,
    ) -> Result<Self> {
        let service = InferenceService::load(&model, source, device, config)?;
        let mut engine = Self::from_client(model, service.client(), search)?;
        engine._service = Some(service);

        Ok(engine)
    }

    /// Builds a persistent search engine over a shared inference client.
    pub fn from_client(
        model: ModelSpec,
        client: InferenceClient,
        search: SearchConfig,
    ) -> Result<Self> {
        let inner = match model.chess_history() {
            Some(ChessHistory::One) => ChessEngineKind::H1(history_mcts(client, search)?),
            Some(ChessHistory::Four) => ChessEngineKind::H4(history_mcts(client, search)?),
            Some(ChessHistory::Eight) => ChessEngineKind::H8(history_mcts(client, search)?),
            None if model.is_chess_classic() => {
                ChessEngineKind::Classic(classic_mcts(client, search)?)
            }
            None => anyhow::bail!("model is not a chess model"),
        };

        Ok(Self {
            _service: None,
            inner,
        })
    }

    pub fn algorithm(&self) -> SearchAlgorithm {
        match &self.inner {
            ChessEngineKind::Classic(mcts) => mcts.algorithm(),
            ChessEngineKind::H1(mcts) => mcts.algorithm(),
            ChessEngineKind::H4(mcts) => mcts.algorithm(),
            ChessEngineKind::H8(mcts) => mcts.algorithm(),
        }
    }

    pub fn select_move(&mut self, game: &ChessGame, request: SearchRequest) -> Result<ChessMove> {
        match &mut self.inner {
            ChessEngineKind::Classic(mcts) => select_classic_move(mcts, game, request),
            ChessEngineKind::H1(mcts) => select_history_move(mcts, game, request),
            ChessEngineKind::H4(mcts) => select_history_move(mcts, game, request),
            ChessEngineKind::H8(mcts) => select_history_move(mcts, game, request),
        }
    }
}

fn classic_mcts(client: InferenceClient, search: SearchConfig) -> Result<ClassicMcts> {
    let evaluator = RepresentedEvaluator::new(ChessClassicRepresentation, client);

    Ok(Mcts::try_new(evaluator, search, ChessRepetitionRules)?)
}

fn history_mcts<const HISTORY: usize>(
    client: InferenceClient,
    search: SearchConfig,
) -> Result<HistoryMcts<HISTORY>> {
    let evaluator = RepresentedEvaluator::new(ChessAzRepresentation::<HISTORY>, client);

    Ok(Mcts::try_new(evaluator, search, ChessRepetitionRules)?)
}

fn select_classic_move(
    mcts: &mut ClassicMcts,
    game: &ChessGame,
    request: SearchRequest,
) -> Result<ChessMove> {
    let result = mcts.search(&game.position(), classic_context(game), request)?;

    Ok(select_result_move(&result, request.mode))
}

fn select_history_move<const HISTORY: usize>(
    mcts: &mut HistoryMcts<HISTORY>,
    game: &ChessGame,
    request: SearchRequest,
) -> Result<ChessMove> {
    let state = ChessAzState::from_game(game);
    let result = mcts.search(&state, game.repetition_context(), request)?;

    Ok(select_result_move(&result, request.mode))
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
    use crate::{Batcher, BatcherConfig, CombinedEncodedBatch, InferenceBackend, ValueHeadSpec};
    use engine_core::notation::GameNotation;
    use games::{chess::ChessUciNotation, ChessRepetitionState};
    use search::{Evaluation, FullGumbelConfig, PositionValue, RootGumbelPuctConfig, SearchBudget};
    use std::path::Path;
    use std::time::Duration;

    struct UniformBackend;

    impl InferenceBackend for UniformBackend {
        fn evaluate(&mut self, batch: &CombinedEncodedBatch) -> Result<Vec<Evaluation>> {
            Ok((0..batch.len())
                .map(|row| {
                    let start = batch.offsets[row] as usize;
                    let end = batch.offsets[row + 1] as usize;

                    Evaluation {
                        logits: vec![0.0; end - start],
                        value: PositionValue::DRAW,
                    }
                })
                .collect())
        }

        fn reload_weights(&mut self, _path: &Path) -> Result<()> {
            Ok(())
        }
    }

    fn inference_service() -> Batcher {
        Batcher::with_backend(
            UniformBackend,
            BatcherConfig {
                preferred_batch_size: 1,
                max_batch_size: 8,
                max_wait: Duration::ZERO,
                max_queued_states: 32,
            },
        )
        .unwrap()
    }

    fn model(history: ChessHistory) -> ModelSpec {
        ModelSpec::chess_se(history, ValueHeadSpec::Scalar { hidden: 8 })
    }

    fn request_for(config: &SearchConfig) -> SearchRequest {
        match config {
            SearchConfig::Puct(_) => SearchRequest::deterministic_puct(1),
            SearchConfig::RootGumbelPuct(_) | SearchConfig::FullGumbel(_) => SearchRequest {
                mode: PolicyMode::Deterministic,
                budget: SearchBudget::Gumbel {
                    simulations: 1,
                    max_considered_actions: 2,
                },
            },
        }
    }

    fn algorithm_for(config: &SearchConfig) -> SearchAlgorithm {
        match config {
            SearchConfig::Puct(_) => SearchAlgorithm::Puct,
            SearchConfig::RootGumbelPuct(_) => SearchAlgorithm::RootGumbelPuct,
            SearchConfig::FullGumbel(_) => SearchAlgorithm::FullGumbel,
        }
    }

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

    #[test]
    fn runtime_preserves_every_representation_and_search_variant() {
        let service = inference_service();
        let models = [
            ModelSpec::chess_classic(1, 8),
            model(ChessHistory::One),
            model(ChessHistory::Four),
            model(ChessHistory::Eight),
        ];
        let searches = [
            SearchConfig::Puct(search::PuctConfig::analysis_default(1)),
            SearchConfig::RootGumbelPuct(RootGumbelPuctConfig::default()),
            SearchConfig::FullGumbel(FullGumbelConfig::default()),
        ];

        for model in models {
            for search in &searches {
                let mut engine = ChessAlphaZeroEngine::from_client(
                    model.clone(),
                    service.client(),
                    search.clone(),
                )
                .unwrap();

                assert_eq!(engine.algorithm(), algorithm_for(search));
                engine
                    .select_move(&ChessGame::default(), request_for(search))
                    .unwrap();
            }
        }
    }

    #[test]
    fn runtime_rejects_a_budget_for_another_algorithm() {
        let service = inference_service();
        let mut engine = ChessAlphaZeroEngine::from_client(
            model(ChessHistory::Four),
            service.client(),
            SearchConfig::FullGumbel(FullGumbelConfig::default()),
        )
        .unwrap();

        let error = engine
            .select_move(&ChessGame::default(), SearchRequest::deterministic_puct(1))
            .unwrap_err();

        assert!(error.downcast_ref::<search::SearchError>().is_some());
        assert!(error.to_string().contains("cannot use"));
    }

    #[test]
    fn runtime_reuses_one_searcher_across_moves() {
        let service = inference_service();
        let mut engine = ChessAlphaZeroEngine::from_client(
            model(ChessHistory::Four),
            service.client(),
            SearchConfig::Puct(search::PuctConfig::analysis_default(1)),
        )
        .unwrap();
        let game = ChessGame::default();
        let searcher = match &engine.inner {
            ChessEngineKind::H4(mcts) => mcts as *const _ as *const (),
            _ => unreachable!("the test constructed an H4 engine"),
        };

        engine
            .select_move(&game, SearchRequest::deterministic_puct(2))
            .unwrap();
        let first = service.stats();

        engine
            .select_move(&game, SearchRequest::deterministic_puct(2))
            .unwrap();
        let second = service.stats();
        let reused_searcher = match &engine.inner {
            ChessEngineKind::H4(mcts) => mcts as *const _ as *const (),
            _ => unreachable!("the test constructed an H4 engine"),
        };

        assert_eq!(reused_searcher, searcher);
        assert!(second.submitted_states > first.submitted_states);
    }
}
