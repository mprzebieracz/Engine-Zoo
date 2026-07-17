#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TerminalValue {
    Win,
    Draw,
    Loss,
}

impl TerminalValue {
    pub const fn as_f32(self) -> f32 {
        match self {
            Self::Win => 1.0,
            Self::Draw => 0.0,
            Self::Loss => -1.0,
        }
    }
}

/// A state that can be explored as a two-player, alternating-turn,
/// zero-sum, perfect-information game tree.
pub trait GameState: Send + 'static {
    type Move: Copy + Eq + std::fmt::Debug + Send + 'static;

    fn initial() -> Self
    where
        Self: Sized;

    fn legal_moves(&self) -> impl Iterator<Item = Self::Move> + '_;

    /// Applies a legal native game move and changes the side to move.
    fn play(&mut self, mv: Self::Move);

    /// Terminal value from the current side-to-move perspective.
    fn terminal_value(&self) -> Option<TerminalValue>;

    fn is_terminal(&self) -> bool {
        self.terminal_value().is_some()
    }
}
