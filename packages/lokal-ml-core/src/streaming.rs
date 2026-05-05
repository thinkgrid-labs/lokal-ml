//! Token stream producer.
//!
//! Wraps the engine's synchronous `chat_stream` callback into an async
//! [`tokio::sync::mpsc`] channel, allowing callers to `.await` each token
//! as it arrives from the inference thread.

use tokio::sync::mpsc;
use crate::engine::{EngineError, LokalEngine};

/// A handle to a running inference stream.
pub struct TokenStream {
    receiver: mpsc::Receiver<String>,
}

impl TokenStream {
    /// Receive the next token from the inference engine.
    ///
    /// Returns `None` when the stream has ended (EOS token received or
    /// `max_tokens` reached).
    pub async fn next(&mut self) -> Option<String> {
        self.receiver.recv().await
    }

    /// Collect all remaining tokens into a single `String`.
    #[must_use]
    pub async fn collect(mut self) -> String {
        let mut result = String::new();
        while let Some(token) = self.next().await {
            result.push_str(&token);
        }
        result
    }
}

/// Spawn a background inference task and return a [`TokenStream`] handle.
///
/// The engine runs on a `tokio::task::spawn_blocking` thread so the async
/// executor is never stalled by CPU-bound matrix math.
#[must_use = "dropping the TokenStream discards all generated tokens"]
pub fn stream_tokens(
    engine: std::sync::Arc<LokalEngine>,
    prompt: String,
    buffer_size: usize,
) -> Result<TokenStream, EngineError> {
    let (tx, rx) = mpsc::channel::<String>(buffer_size);

    tokio::task::spawn_blocking(move || {
        let result = engine.chat_stream(&prompt, move |token| {
            // Ignore send errors — the receiver may have been dropped
            let _ = tx.blocking_send(token.to_string());
        });

        if let Err(e) = result {
            tracing::error!("Inference error: {}", e);
        }
    });

    Ok(TokenStream { receiver: rx })
}

#[cfg(test)]
mod tests {
    use crate::engine::{EngineConfig, EngineError, LokalEngine};
    use std::path::Path;
    use tempfile::tempdir;

    #[tokio::test]
    async fn stream_errors_on_invalid_model() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, b"not a gguf").unwrap();
        // Loading should fail; stream_tokens never gets a chance to run.
        let result = LokalEngine::load(&path, EngineConfig::default());
        assert!(matches!(result, Err(EngineError::LoadFailed { .. })));
    }

    #[test]
    fn stream_errors_on_missing_model() {
        let result = LokalEngine::load(Path::new("/no/such/model.gguf"), EngineConfig::default());
        assert!(matches!(result, Err(EngineError::LoadFailed { .. })));
    }
}
