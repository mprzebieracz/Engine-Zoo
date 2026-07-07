use crate::agent::{Agent, PolicyMode};
use crate::game::{Action, Game};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct SessionConfig {
    pub max_moves: usize,
    pub opening_moves: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        SessionConfig {
            max_moves: 512,
            opening_moves: 0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MoveEvent {
    pub ply: usize,
    pub side: usize,
    pub action: Action,
    pub mv: String,
    pub board: String,
    pub terminal: bool,
    pub reward: f32,
}

pub fn play_agents<G: Game>(
    mut game: G,
    first: &mut impl Agent<G>,
    second: &mut impl Agent<G>,
    cfg: &SessionConfig,
) -> Vec<MoveEvent> {
    let mut events = Vec::new();
    let mut ply = 0usize;
    while !game.is_terminal() && ply < cfg.max_moves {
        let mode = if ply < cfg.opening_moves {
            PolicyMode::Explore
        }
        else {
            PolicyMode::Deterministic
        };
        let side = ply % 2;
        let action = if side == 0 {
            first.act_with_mode(&game, mode)
        }
        else {
            second.act_with_mode(&game, mode)
        };
        let mv = game.format_action(action);
        game.step(action);
        events.push(MoveEvent {
            ply,
            side,
            action,
            mv,
            board: game.to_string(),
            terminal: game.is_terminal(),
            reward: game.reward(),
        });
        ply += 1;
    }
    events
}
