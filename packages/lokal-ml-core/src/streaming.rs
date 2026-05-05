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
    use super::*;
    use crate::engine::{EngineConfig, LokalEngine};
    use std::sync::Arc;
    use tempfile::tempdir;

    #[tokio::test]
    async fn stream_collects_all_tokens() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("stub.gguf");
        std::fs::write(&model_path, b"stub").unwrap();

        let engine = Arc::new(LokalEngine::load(&model_path, EngineConfig::default()).unwrap());
        let mut stream = stream_tokens(engine, "test prompt".to_string(), 32).unwrap();

        let collected = stream.collect().await;
        assert!(!collected.is_empty());
    }
}
