use crate::agent::{Agent, PolicyMode};
use crate::game::Game;

#[derive(Clone, Debug)]
pub struct ArenaConfig {
    pub games: usize,
    /// Games longer than this are scored as draws.
    pub max_moves: usize,
    /// Moves per game chosen with `PolicyMode::Explore`, so repeated matchups
    /// between deterministic agents don't all replay one game.
    pub opening_moves: usize,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        ArenaConfig {
            games: 40,
            max_moves: 512,
            opening_moves: 6,
        }
    }
}

/// Pits `candidate` against `baseline` and returns the candidate's score
/// Colors alternate every game.
pub fn evaluate<G: Game>(
    candidate: &mut impl Agent<G>,
    baseline: &mut impl Agent<G>,
    cfg: &ArenaConfig,
) -> f32 {
    let mut score = 0.0;
    for i in 0..cfg.games {
        if i.is_multiple_of(2) {
            score += play_single_game::<G>(candidate, baseline, cfg);
        }
        else {
            score += 1.0 - play_single_game::<G>(baseline, candidate, cfg);
        }
        println!(
            "arena: {}/{} games, candidate score {score:.1}",
            i + 1,
            cfg.games
        );
    }
    score / cfg.games as f32
}

/// Plays one game; returns the first player's score (1 / 0.5 / 0).
fn play_single_game<'a, G: Game>(
    first: &'a mut dyn Agent<G>,
    second: &'a mut dyn Agent<G>,
    cfg: &ArenaConfig,
) -> f32 {
    let mut game = G::default();
    let mut move_idx = 0usize;

    while !game.is_terminal() && move_idx < cfg.max_moves {
        let agent = if move_idx.is_multiple_of(2) {
            &mut *first
        }
        else {
            &mut *second
        };

        let mode = if move_idx < cfg.opening_moves {
            PolicyMode::Explore
        }
        else {
            PolicyMode::Deterministic
        };

        let action = agent.act_with_mode(&game, mode);

        game.step(action);
        move_idx += 1;
    }

    if !game.is_terminal() || game.reward() == 0.0 {
        return 0.5;
    }
    // Decisive terminal rewards are +1 for the side that made the last move.
    if (move_idx - 1).is_multiple_of(2) {
        1.0
    }
    else {
        0.0
    }
}
