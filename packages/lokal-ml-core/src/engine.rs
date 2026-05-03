//! GGUF inference engine.
//!
//! Wraps the `llama-cpp-2` crate to load a quantized model from disk and
//! execute inference with token streaming. The engine is designed to run on
//! a dedicated background thread, keeping the UI thread fully responsive.

use std::path::Path;
use thiserror::Error;
use tracing::{debug, info};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Failed to load model from {path}: {reason}")]
    LoadFailed { path: String, reason: String },

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Model not loaded — call LokalEngine::load() first")]
    NotLoaded,
}

/// Configuration for the inference engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Context window size in tokens (default: 2048)
    pub context_size: u32,
    /// Number of CPU threads to use for inference
    pub threads: u32,
    /// Maximum tokens to generate per response
    pub max_tokens: u32,
    /// Sampling temperature (0.0 = deterministic, 1.0 = creative)
    pub temperature: f32,
    /// Enable Metal GPU acceleration on Apple Silicon (iOS/macOS)
    pub use_metal: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            context_size: 2048,
            threads: 4,
            max_tokens: 512,
            temperature: 0.7,
            use_metal: cfg!(target_os = "ios") || cfg!(target_os = "macos"),
        }
    }
}

/// A loaded inference engine instance.
///
/// This is intentionally opaque — the llama-cpp-2 model handle is held
/// internally. Thread-safety is enforced by the underlying C++ layer.
pub struct LokalEngine {
    config: EngineConfig,
    model_path: String,
    // NOTE: The actual llama.cpp model/context handles will be added here
    // once the llama-cpp-2 crate API is finalised.
    // model: llama_cpp_2::model::LlamaModel,
}

impl LokalEngine {
    /// Load a GGUF model from disk and initialise the inference context.
    ///
    /// This is a blocking operation and should be called from a background thread.
    pub fn load(model_path: &Path, config: EngineConfig) -> Result<Self, EngineError> {
        info!("Loading model from {:?}", model_path);

        if !model_path.exists() {
            return Err(EngineError::LoadFailed {
                path: model_path.display().to_string(),
                reason: "File does not exist".to_string(),
            });
        }

        // TODO: Initialise llama_cpp_2::model::LlamaModel here when integrating
        // the llama-cpp-2 crate. The placeholder below validates the file path
        // and config without touching the C++ layer during initial development.

        info!(
            context_size = config.context_size,
            threads = config.threads,
            use_metal = config.use_metal,
            "Engine initialised"
        );

        Ok(Self {
            config,
            model_path: model_path.display().to_string(),
        })
    }

    /// Execute a chat prompt and stream tokens via the provided callback.
    ///
    /// `on_token` is called on the **inference thread** for each generated token.
    /// The caller is responsible for dispatching to the UI thread (via JSI/FFI).
    pub fn chat_stream(
        &self,
        prompt: &str,
        on_token: impl Fn(&str) + Send + 'static,
    ) -> Result<(), EngineError> {
        debug!(prompt_len = prompt.len(), "Starting inference stream");

        // TODO: Wire in llama_cpp_2 inference loop here.
        // The placeholder emits a single echo token so the streaming pipeline
        // can be tested end-to-end before the C++ layer is wired up.
        on_token("[STUB] Model loaded from: ");
        on_token(&self.model_path);
        on_token(" | Prompt: ");
        on_token(prompt);

        Ok(())
    }

    /// Return the model path this engine was loaded from.
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Return the active engine configuration.
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn load_fails_for_missing_file() {
        let result = LokalEngine::load(Path::new("/nonexistent/model.gguf"), Default::default());
        assert!(matches!(result, Err(EngineError::LoadFailed { .. })));
    }

    #[test]
    fn chat_stream_emits_tokens() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("stub.gguf");
        std::fs::write(&model_path, b"stub").unwrap();

        let engine = LokalEngine::load(&model_path, Default::default()).unwrap();

        let mut tokens = Vec::new();
        engine
            .chat_stream("Hello", |tok| tokens.push(tok.to_string()))
            .unwrap();

        assert!(!tokens.is_empty());
    }
}
