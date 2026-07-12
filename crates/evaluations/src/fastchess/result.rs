//! Parsing of normalized Fastchess match results.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Wdl {
    pub wins: u32,
    pub draws: u32,
    pub losses: u32,
}

impl Wdl {
    pub fn games(self) -> u32 {
        self.wins + self.draws + self.losses
    }
    pub fn score(self) -> f64 {
        (self.wins as f64 + self.draws as f64 * 0.5) / self.games().max(1) as f64
    }
}

pub fn parse_wdl(text: &str) -> Option<Wdl> {
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("wins") && lower.contains("draw") && lower.contains("loss") {
            if let (Some(wins), Some(draws), Some(losses)) = (
                labeled_count(line, "wins"),
                labeled_count(line, "draws"),
                labeled_count(line, "losses"),
            ) {
                return Some(Wdl {
                    wins,
                    draws,
                    losses,
                });
            }
        }
        if lower.contains("score of") {
            let nums: Vec<u32> = line
                .split(|c: char| !c.is_ascii_digit())
                .filter_map(|s| s.parse().ok())
                .collect();
            if nums.len() >= 3 {
                return Some(Wdl {
                    wins: nums[0],
                    draws: nums[2],
                    losses: nums[1],
                });
            }
        }
    }
    None
}

pub fn parse_score(text: &str) -> Option<f64> {
    for line in text.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.contains("score") {
            for token in line.split(|c: char| c.is_whitespace() || c == '[' || c == ']') {
                let percent = token.ends_with('%');
                if let Ok(value) = token.trim_end_matches('%').parse::<f64>() {
                    if (0.0..=1.0).contains(&value) {
                        return Some(value);
                    }
                    if (0.0..=100.0).contains(&value) && percent {
                        return Some(value / 100.0);
                    }
                }
            }
        }
    }
    None
}

fn labeled_count(line: &str, label: &str) -> Option<u32> {
    let lower = line.to_ascii_lowercase();
    let start = lower.find(label)? + label.len();
    line[start..]
        .split(|c: char| !c.is_ascii_digit())
        .find_map(|token| token.parse().ok())
}
