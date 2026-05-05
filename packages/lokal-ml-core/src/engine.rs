use std::num::NonZeroU32;
use std::path::Path;

use llama_cpp_2::{
    context::params::LlamaContextParams,
    llama_batch::LlamaBatch,
    model::{params::LlamaModelParams, AddBos, LlamaModel},
    sampling::LlamaSampler,
};
use thiserror::Error;
use tracing::{debug, info};

use crate::backend;

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
    /// Sampling temperature (0.0 = greedy / deterministic, 1.0 = creative)
    pub temperature: f32,
    /// Number of model layers to offload to GPU.
    /// `u32::MAX` offloads all layers (Metal on Apple Silicon, Vulkan on Android).
    /// `0` forces CPU-only execution.
    pub gpu_layers: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            context_size: 2048,
            threads: 4,
            max_tokens: 512,
            temperature: 0.7,
            // Offload all layers on Apple Silicon; CPU-only elsewhere.
            gpu_layers: if cfg!(any(target_os = "ios", target_os = "macos")) {
                u32::MAX
            } else {
                0
            },
        }
    }
}

/// A loaded inference engine instance.
pub struct LokalEngine {
    config: EngineConfig,
    model_path: String,
    model: LlamaModel,
}

// LlamaModel is Send + Sync (llama.cpp handles its own internal locking).
// Declaring these explicitly documents the invariant for future contributors.
unsafe impl Send for LokalEngine {}
unsafe impl Sync for LokalEngine {}

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

        let model_params = LlamaModelParams::default().with_n_gpu_layers(config.gpu_layers);

        let model = LlamaModel::load_from_file(backend::get(), model_path, &model_params)
            .map_err(|e| EngineError::LoadFailed {
                path: model_path.display().to_string(),
                reason: e.to_string(),
            })?;

        info!(
            context_size = config.context_size,
            threads = config.threads,
            gpu_layers = config.gpu_layers,
            n_embd = model.n_embd(),
            n_layers = model.n_layer(),
            "Engine initialised"
        );

        Ok(Self {
            config,
            model_path: model_path.display().to_string(),
            model,
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

        let n_threads = i32::try_from(self.config.threads).unwrap_or(4);

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(self.config.context_size))
            .with_n_threads(n_threads)
            .with_n_threads_batch(n_threads);

        let mut ctx = self
            .model
            .new_context(backend::get(), ctx_params)
            .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;

        if tokens.is_empty() {
            return Ok(());
        }

        let n_prompt = tokens.len();
        if n_prompt >= self.config.context_size as usize {
            return Err(EngineError::InferenceFailed(format!(
                "Prompt is too long ({n_prompt} tokens) for context window ({})",
                self.config.context_size
            )));
        }

        // Prefill: decode the entire prompt, enabling logits only for the last token.
        let mut batch = LlamaBatch::new(n_prompt, 1);
        for (i, &token) in tokens.iter().enumerate() {
            let is_last = i == n_prompt - 1;
            batch
                .add(token, i as i32, &[0_i32], is_last)
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;

        // Autoregressive generation.
        let sampler_chain: Vec<LlamaSampler> = if self.config.temperature == 0.0 {
            vec![LlamaSampler::greedy()]
        } else {
            vec![
                LlamaSampler::temp(self.config.temperature),
                LlamaSampler::dist(1337),
            ]
        };
        let mut sampler = LlamaSampler::chain_simple(sampler_chain);
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut pos = n_prompt as i32;
        let mut n_generated: u32 = 0;

        loop {
            // Sample from last decode's logits (idx = -1 → last output position).
            let token = sampler.sample(&ctx, -1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                break;
            }

            let piece = self
                .model
                .token_to_piece(token, &mut decoder, true, None)
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;

            on_token(&piece);
            n_generated += 1;

            if n_generated >= self.config.max_tokens {
                break;
            }

            // Feed the sampled token back so the next sample has fresh logits.
            batch.clear();
            batch
                .add(token, pos, &[0_i32], true)
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;
            pos += 1;

            ctx.decode(&mut batch)
                .map_err(|e| EngineError::InferenceFailed(e.to_string()))?;
        }

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
    fn load_fails_for_invalid_gguf() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, b"this is not a gguf file").unwrap();
        // llama.cpp will reject the file — expect a LoadFailed error.
        let result = LokalEngine::load(&path, Default::default());
        assert!(matches!(result, Err(EngineError::LoadFailed { .. })));
    }
}
