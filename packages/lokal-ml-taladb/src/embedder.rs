//! Embedding model runner.
//!
//! Loads `all-MiniLM-L6-v2` in quantized GGUF format (~22 MB) and converts
//! text chunks into 384-dimensional float vectors suitable for cosine similarity
//! search in TalaDB's HNSW index.
//!
//! The embedding model is downloaded silently on first [`Embedder::new`] call
//! if not already cached locally.

use thiserror::Error;
use tracing::{info, warn};

/// Errors returned by the embedder.
#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("Embedding model not found at {path}")]
    ModelNotFound { path: String },

    #[error("Embedding generation failed: {0}")]
    InferenceFailed(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Model ID used in the registry for the embedding model.
pub const EMBED_MODEL_ID: &str = "all-minilm-l6-v2";

/// Dimensionality of the all-MiniLM-L6-v2 output vectors.
pub const EMBED_DIM: usize = 384;

/// A loaded embedding model instance.
pub struct Embedder {
    model_path: String,
    // TODO: llama_cpp_2 model handle for the embedding model
}

impl Embedder {
    /// Load the embedding model from `model_path`.
    ///
    /// This should be called once and the instance reused across all chunks
    /// in a batch — model loading is expensive.
    pub fn new(model_path: &std::path::Path) -> Result<Self, EmbedError> {
        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound {
                path: model_path.display().to_string(),
            });
        }

        info!("Loading embedding model from {:?}", model_path);

        // TODO: Initialise llama_cpp_2 with embedding mode enabled
        // (llama_cpp_2::model::params::LlamaModelParams with embedding = true)

        Ok(Self {
            model_path: model_path.display().to_string(),
        })
    }

    /// Generate a 384-dimensional embedding vector for the given text.
    ///
    /// Returns a stub vector of zeros during development — wire in the real
    /// llama.cpp embedding call when the C++ layer is integrated.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            warn!("embed() called with empty text — returning zero vector");
            return Ok(vec![0.0f32; EMBED_DIM]);
        }

        // TODO: Replace with actual llama.cpp embedding inference:
        // 1. Tokenize `text` using the model's vocabulary
        // 2. Run a forward pass with llama_get_embeddings()
        // 3. Return the normalised mean-pool of the last hidden state

        // Stub: deterministic pseudo-vector based on text length
        let stub: Vec<f32> = (0..EMBED_DIM)
            .map(|i| ((text.len() as f32 + i as f32) % 100.0) / 100.0)
            .collect();

        Ok(stub)
    }

    /// Embed a batch of texts, returning one vector per input.
    ///
    /// Currently calls `embed()` sequentially. When the real llama.cpp layer is
    /// integrated this should be replaced with a single-pass forward call using
    /// `llama_batch` so tokenisation cost is amortised across the whole batch.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn embed_returns_correct_dim() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("embed.gguf");
        std::fs::write(&path, b"stub").unwrap();

        let embedder = Embedder::new(&path).unwrap();
        let vec = embedder.embed("hello world").unwrap();
        assert_eq!(vec.len(), EMBED_DIM);
    }

    #[test]
    fn embed_batch_consistent_with_single() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("embed.gguf");
        std::fs::write(&path, b"stub").unwrap();

        let embedder = Embedder::new(&path).unwrap();
        let single = embedder.embed("test").unwrap();
        let batch = embedder.embed_batch(&["test"]).unwrap();
        assert_eq!(single, batch[0]);
    }
}
