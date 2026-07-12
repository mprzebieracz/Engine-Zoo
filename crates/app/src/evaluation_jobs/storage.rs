use anyhow::Result;
use serde::Deserialize;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use super::Job;

pub(super) fn artifact_names(dir: &Path) -> Result<Vec<String>> {
    Ok(fs::read_dir(dir)?
        .flatten()
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(|kind| kind.is_file())
                .map(|_| entry.file_name().to_string_lossy().into_owned())
        })
        .filter(|name| safe_name(name))
        .collect())
}

pub(super) fn write_job(dir: &Path, job: &Job) -> Result<()> {
    let tmp = dir.join("job.json.tmp");
    fs::write(&tmp, serde_json::to_vec_pretty(job)?)?;
    fs::rename(tmp, dir.join("job.json"))?;
    Ok(())
}

pub(super) fn read_json<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

pub(super) fn safe_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && !name.contains('/') && !name.contains('\\')
}

pub(super) fn validate_id(id: &str) -> Result<()> {
    anyhow::ensure!(
        safe_name(id)
            && id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'),
        "invalid job id"
    );
    Ok(())
}

pub(super) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_names_are_pathsafe() {
        assert!(safe_name("games.pgn"));
        assert!(!safe_name("../secret"));
        assert!(!safe_name("a/b"));
    }

    #[test]
    fn ids_are_restricted() {
        assert!(validate_id("123-abc").is_ok());
        assert!(validate_id("../x").is_err());
    }

    #[test]
    fn rejects_platform_separators_and_ambiguous_names() {
        for name in ["", ".", "..", "a/b", r"a\b"] {
            assert!(!safe_name(name), "{name:?}");
        }
        for id in ["with space", "unicode-λ", "name.json", "a/b"] {
            assert!(validate_id(id).is_err(), "{id:?}");
        }
    }
}
