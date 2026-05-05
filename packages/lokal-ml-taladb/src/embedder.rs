use std::num::NonZeroU32;

use llama_cpp_2::{
    context::params::{LlamaContextParams, LlamaPoolingType},
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
};
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

/// Expected output dimensionality for all-MiniLM-L6-v2.
pub const EMBED_DIM: usize = 384;

/// A loaded embedding model instance.
pub struct Embedder {
    model_path: String,
    model: LlamaModel,
    /// Embedding dimension as reported by the model (should equal EMBED_DIM).
    n_embd: usize,
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

        // Embedding models are small enough to run on CPU without quality loss.
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);

        let model =
            LlamaModel::load_from_file(lokal_ml_core::backend::get(), model_path, &model_params)
                .map_err(|e| EmbedError::InferenceFailed(e.to_string()))?;

        let n_embd = model.n_embd() as usize;

        info!(n_embd, "Embedding model loaded");

        Ok(Self {
            model_path: model_path.display().to_string(),
            model,
            n_embd,
        })
    }

    /// Generate a normalised embedding vector for the given text.
    ///
    /// Uses mean-pool (LLAMA_POOLING_TYPE_MEAN) which is what all-MiniLM-L6-v2
    /// was trained with. The output is L2-normalised so cosine similarity in
    /// TalaDB reduces to a dot product.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if text.is_empty() {
            warn!("embed() called with empty text — returning zero vector");
            return Ok(vec![0.0f32; self.n_embd]);
        }

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(512))
            .with_n_threads(2)
            .with_n_threads_batch(2)
            .with_embeddings(true)
            .with_pooling_type(LlamaPoolingType::Mean);

        let mut ctx = self
            .model
            .new_context(lokal_ml_core::backend::get(), ctx_params)
            .map_err(|e| EmbedError::InferenceFailed(e.to_string()))?;

        let tokens = self
            .model
            .str_to_token(text, AddBos::Always)
            .map_err(|e| EmbedError::InferenceFailed(e.to_string()))?;

        let n = tokens.len().max(1);
        let mut batch = LlamaBatch::new(n, 1);

        for (i, &token) in tokens.iter().enumerate() {
            batch
                .add(token, i as i32, &[0_i32], true)
                .map_err(|e| EmbedError::InferenceFailed(e.to_string()))?;
        }

        ctx.encode(&mut batch)
            .map_err(|e| EmbedError::InferenceFailed(e.to_string()))?;

        let raw = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| EmbedError::InferenceFailed(e.to_string()))?;

        // L2-normalise so cosine similarity == dot product in TalaDB's HNSW index.
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        Ok(raw.iter().map(|x| x / norm).collect())
    }

    /// Embed a batch of texts, returning one vector per input.
    ///
    /// Calls `embed()` sequentially. A future optimisation is to batch all
    /// tokens into a single `llama_batch` to amortise the forward-pass cost.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn new_fails_for_missing_model() {
        let result = Embedder::new(std::path::Path::new("/no/such/model.gguf"));
        assert!(matches!(result, Err(EmbedError::ModelNotFound { .. })));
    }

    #[test]
    fn new_fails_for_invalid_gguf() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, b"not a gguf file").unwrap();
        let result = Embedder::new(&path);
        assert!(matches!(result, Err(EmbedError::InferenceFailed(_))));
    }
}
