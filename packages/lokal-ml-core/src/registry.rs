//! Model registry manifest.
//!
//! Parses the `registry/models.json` manifest that maps model IDs (e.g.
//! `"gemma-2b-int4"`) to download URLs, SHA-256 hashes, sizes, and minimum
//! hardware requirements.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RegistryError {
    #[error("Model '{0}' not found in registry")]
    ModelNotFound(String),
    #[error("Failed to parse registry manifest: {0}")]
    ParseError(#[from] serde_json::Error),
}

/// Specification for a single model entry in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Stable model identifier (e.g. "gemma-2b-int4")
    pub id: String,
    /// Direct download URL for the `.gguf` file
    pub url: String,
    /// SHA-256 hex digest for integrity verification
    pub sha256: String,
    /// Uncompressed file size in bytes (used for progress reporting)
    pub size_bytes: u64,
    /// Minimum device RAM required in MB
    pub min_ram_mb: u64,
}

/// The full registry manifest as deserialised from `models.json`.
#[derive(Debug, Deserialize)]
pub struct Registry {
    pub models: HashMap<String, ModelSpec>,
}

impl Registry {
    /// Parse a registry manifest from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, RegistryError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Look up a model by its stable ID.
    pub fn get(&self, model_id: &str) -> Result<&ModelSpec, RegistryError> {
        self.models
            .get(model_id)
            .ok_or_else(|| RegistryError::ModelNotFound(model_id.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
    {
      "models": {
        "gemma-2b-int4": {
          "id": "gemma-2b-int4",
          "url": "https://huggingface.co/google/gemma-2b/resolve/main/gemma-2b-int4.gguf",
          "sha256": "deadbeef",
          "size_bytes": 1610612736,
          "min_ram_mb": 2200
        }
      }
    }"#;

    #[test]
    fn parses_manifest() {
        let registry = Registry::from_json(SAMPLE).unwrap();
        let spec = registry.get("gemma-2b-int4").unwrap();
        assert_eq!(spec.min_ram_mb, 2200);
    }

    #[test]
    fn returns_error_for_unknown_model() {
        let registry = Registry::from_json(SAMPLE).unwrap();
        assert!(matches!(
            registry.get("nonexistent"),
            Err(RegistryError::ModelNotFound(_))
        ));
    }
}
