//! In-process matches between two engine agents.

use engine_core::{Agent, GameState, PolicyMode, TerminalValue};
use std::error::Error;
use std::fmt::{self, Display, Formatter};

#[derive(Clone, Debug)]
pub struct ArenaConfig {
    pub games: usize,
    /// Games longer than this are scored as draws.
    pub max_moves: usize,
    /// Initial moves selected using exploratory policy.
    pub opening_moves: usize,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            games: 40,
            max_moves: 512,
            opening_moves: 6,
        }
    }
}

impl ArenaConfig {
    pub fn validate(&self) -> Result<(), ArenaConfigError> {
        if self.games == 0 {
            return Err(ArenaConfigError);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArenaConfigError;

impl Display for ArenaConfigError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str("arena must contain at least one game")
    }
}

impl Error for ArenaConfigError {}

/// Candidate-relative aggregate for a completed match.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ArenaResult {
    pub wins: usize,
    pub draws: usize,
    pub losses: usize,
}

impl ArenaResult {
    pub fn games(self) -> usize {
        self.wins + self.draws + self.losses
    }

    /// Candidate score in `[0, 1]`, with draws worth half a point.
    pub fn score(self) -> f32 {
        (self.wins as f32 + self.draws as f32 * 0.5) / self.games() as f32
    }

    fn record(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Win => self.wins += 1,
            Outcome::Draw => self.draws += 1,
            Outcome::Loss => self.losses += 1,
        }
    }
}

/// Pits `candidate` against `baseline`, alternating colors every game.
pub fn evaluate<G: GameState + Default>(
    candidate: &mut impl Agent<G>,
    baseline: &mut impl Agent<G>,
    config: &ArenaConfig,
) -> Result<ArenaResult, ArenaConfigError> {
    config.validate()?;
    let mut result = ArenaResult::default();

    for game_index in 0..config.games {
        let outcome = if game_index.is_multiple_of(2) {
            play_single_game::<G, _, _>(candidate, baseline, config)
        } else {
            play_single_game::<G, _, _>(baseline, candidate, config).reverse()
        };
        result.record(outcome);
        println!(
            "arena: {}/{} games, candidate score {:.1}",
            game_index + 1,
            config.games,
            result.wins as f32 + result.draws as f32 * 0.5
        );
    }

    Ok(result)
}

#[derive(Clone, Copy)]
enum Outcome {
    Win,
    Draw,
    Loss,
}

impl Outcome {
    fn reverse(self) -> Self {
        match self {
            Self::Win => Self::Loss,
            Self::Draw => Self::Draw,
            Self::Loss => Self::Win,
        }
    }
}

fn play_single_game<G, A, B>(first: &mut A, second: &mut B, config: &ArenaConfig) -> Outcome
where
    G: GameState + Default,
    A: Agent<G>,
    B: Agent<G>,
{
    let mut game = G::default();
    let mut move_index = 0;
    while !game.is_terminal() && move_index < config.max_moves {
        let mode = if move_index < config.opening_moves {
            PolicyMode::Explore
        } else {
            PolicyMode::Deterministic
        };
        let mv = if move_index.is_multiple_of(2) {
            first.select_move(&game, mode)
        } else {
            second.select_move(&game, mode)
        };
        game.play(mv);
        move_index += 1;
    }

    if !game.is_terminal() {
        return Outcome::Draw;
    }

    match game.terminal_value() {
        None | Some(TerminalValue::Draw) => Outcome::Draw,
        Some(TerminalValue::Loss) if (move_index - 1).is_multiple_of(2) => Outcome::Win,
        Some(TerminalValue::Loss) => Outcome::Loss,
        Some(TerminalValue::Win) if (move_index - 1).is_multiple_of(2) => Outcome::Loss,
        Some(TerminalValue::Win) => Outcome::Win,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use games::connect4::Connect4Move;

    #[derive(Clone, Default)]
    struct TinyGame(Vec<Connect4Move>);

    impl Display for TinyGame {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.0.len())
        }
    }

    impl GameState for TinyGame {
        type Move = Connect4Move;
        fn initial() -> Self {
            Self::default()
        }
        fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_ {
            (0..4).map(|i| Connect4Move::new(i).unwrap())
        }
        fn play(&mut self, mv: Self::Move) {
            self.0.push(mv);
        }
        fn is_terminal(&self) -> bool {
            self.0.last().is_some_and(|action| action.column() != 0)
        }
        fn terminal_value(&self) -> Option<TerminalValue> {
            self.0.last().map(|mv| {
                if mv.column() == 3 {
                    TerminalValue::Draw
                } else {
                    TerminalValue::Loss
                }
            })
        }
    }

    struct RecordingAgent {
        action: Connect4Move,
        modes: Vec<PolicyMode>,
    }
    impl RecordingAgent {
        fn new(column: u8) -> Self {
            Self {
                action: Connect4Move::new(column).unwrap(),
                modes: Vec::new(),
            }
        }
    }
    impl Agent<TinyGame> for RecordingAgent {
        fn select_move(&mut self, _: &TinyGame, mode: PolicyMode) -> Connect4Move {
            self.modes.push(mode);
            self.action
        }
    }

    fn config(games: usize) -> ArenaConfig {
        ArenaConfig {
            games,
            max_moves: 4,
            opening_moves: 0,
        }
    }

    #[test]
    fn alternates_colors_and_aggregates_wins_and_losses() {
        let mut candidate = RecordingAgent::new(1);
        let mut baseline = RecordingAgent::new(2);
        let result = evaluate::<TinyGame>(&mut candidate, &mut baseline, &config(2)).unwrap();
        assert_eq!(
            result,
            ArenaResult {
                wins: 1,
                draws: 0,
                losses: 1
            }
        );
        assert_eq!(candidate.modes.len(), 1);
        assert_eq!(baseline.modes.len(), 1);
    }

    #[test]
    fn handles_terminal_and_move_limit_draws() {
        let mut draw_agent = RecordingAgent::new(3);
        let mut opponent = RecordingAgent::new(2);
        assert_eq!(
            evaluate::<TinyGame>(&mut draw_agent, &mut opponent, &config(1))
                .unwrap()
                .draws,
            1
        );

        let mut first = RecordingAgent::new(0);
        let mut second = RecordingAgent::new(0);
        assert_eq!(
            evaluate::<TinyGame>(&mut first, &mut second, &config(1))
                .unwrap()
                .draws,
            1
        );
        assert_eq!((first.modes.len(), second.modes.len()), (2, 2));
    }

    #[test]
    fn explores_opening_then_plays_deterministically() {
        let cfg = ArenaConfig {
            games: 1,
            max_moves: 4,
            opening_moves: 2,
        };
        let mut first = RecordingAgent::new(0);
        let mut second = RecordingAgent::new(0);
        evaluate::<TinyGame>(&mut first, &mut second, &cfg).unwrap();
        assert_eq!(
            first.modes,
            [PolicyMode::Explore, PolicyMode::Deterministic]
        );
        assert_eq!(
            second.modes,
            [PolicyMode::Explore, PolicyMode::Deterministic]
        );
    }

    #[test]
    fn validates_games_and_aggregates_score() {
        assert_eq!(
            config(0).validate().unwrap_err().to_string(),
            "arena must contain at least one game"
        );
        let result = ArenaResult {
            wins: 2,
            draws: 1,
            losses: 1,
        };
        assert_eq!(result.games(), 4);
        assert_eq!(result.score(), 0.625);
    }
}
