use engine_core::game::TerminalValue;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::fmt;

/// A value from the player-to-move perspective of the state or node that
/// stores it. `WIN`, `DRAW`, and `LOSS` are respectively `1`, `0`, and `-1`.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct PositionValue(f32);

/// Returned when a position value is non-finite or outside the closed unit
/// interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidValue;

impl fmt::Display for InvalidValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("position value must be finite and in [-1, 1]")
    }
}

impl Error for InvalidValue {}

impl PositionValue {
    pub const LOSS: Self = Self(-1.0);
    pub const DRAW: Self = Self(0.0);
    pub const WIN: Self = Self(1.0);

    pub fn new(value: f32) -> Result<Self, InvalidValue> {
        if value.is_finite() && (-1.0..=1.0).contains(&value) {
            Ok(Self(value))
        } else {
            Err(InvalidValue)
        }
    }

    /// Converts a non-fallible model output into a legal search value.
    /// Finite values are clamped to the closed unit interval; `NaN` becomes
    /// neutral because the current evaluator trait cannot return an error.
    pub fn new_clamped(value: f32) -> Self {
        Self(if value.is_nan() {
            0.0
        } else {
            value.clamp(-1.0, 1.0)
        })
    }

    pub const fn as_f32(self) -> f32 {
        self.0
    }

    pub const fn flipped(self) -> Self {
        Self(-self.0)
    }
}

impl From<TerminalValue> for PositionValue {
    fn from(value: TerminalValue) -> Self {
        match value {
            TerminalValue::Win => Self::WIN,
            TerminalValue::Draw => Self::DRAW,
            TerminalValue::Loss => Self::LOSS,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_clamps_and_flips_values() {
        assert_eq!(PositionValue::new(1.0), Ok(PositionValue::WIN));
        assert_eq!(PositionValue::new(-1.1), Err(InvalidValue));
        assert_eq!(PositionValue::new(f32::NAN), Err(InvalidValue));
        assert_eq!(
            PositionValue::new_clamped(f32::INFINITY),
            PositionValue::WIN
        );
        assert_eq!(
            PositionValue::new_clamped(f32::NEG_INFINITY),
            PositionValue::LOSS
        );
        assert_eq!(PositionValue::new_clamped(f32::NAN), PositionValue::DRAW);
        assert_eq!(PositionValue::WIN.flipped(), PositionValue::LOSS);
    }
}
