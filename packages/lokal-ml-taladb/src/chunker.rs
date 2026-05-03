//! Text chunker for the RAG ingestion pipeline.
//!
//! Splits raw text or Markdown into fixed-size windows (measured in
//! whitespace-delimited words as a proxy for tokens) with configurable
//! overlap, suitable for embedding and vector search.
//!
//! ## Example
//! ```rust
//! use lokal_ml_taladb::chunker::chunk_text;
//!
//! let chunks = chunk_text("Hello world. This is a test.", "doc1", 3, 1).unwrap();
//! assert_eq!(chunks.len(), 3);
//! ```

use thiserror::Error;
use ulid::Ulid;

#[derive(Debug, Error)]
pub enum ChunkError {
    #[error("chunk_size must be > 0")]
    InvalidChunkSize,
    #[error("overlap must be less than chunk_size")]
    InvalidOverlap,
}

/// A single text chunk produced by the chunker.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Stable unique ID for this chunk (ULID)
    pub id: String,
    /// The original document ID this chunk belongs to
    pub doc_id: String,
    /// The chunk index within the document (0-based)
    pub index: usize,
    /// The raw text content of this chunk
    pub text: String,
    /// Approximate token count (word count proxy)
    pub token_count: usize,
}

/// Split `text` into overlapping windows of `chunk_size` words with `overlap`
/// words of context carried forward from the previous window.
///
/// Uses whitespace-delimited words as a fast proxy for tokens. For production
/// accuracy, replace with `tiktoken-rs` BPE counting.
pub fn chunk_text(
    text: &str,
    doc_id: &str,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<Chunk>, ChunkError> {
    if chunk_size == 0 {
        return Err(ChunkError::InvalidChunkSize);
    }
    if overlap >= chunk_size {
        return Err(ChunkError::InvalidOverlap);
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() {
        return Ok(vec![]);
    }

    let step = chunk_size - overlap;
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut index = 0;

    while start < words.len() {
        let end = (start + chunk_size).min(words.len());
        let chunk_words = &words[start..end];
        let chunk_text = chunk_words.join(" ");
        let token_count = chunk_words.len();

        chunks.push(Chunk {
            id: Ulid::new().to_string(),
            doc_id: doc_id.to_string(),
            index,
            text: chunk_text,
            token_count,
        });

        if end == words.len() {
            break;
        }

        start += step;
        index += 1;
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_short_text() {
        let text = "one two three four five six seven eight nine ten";
        let chunks = chunk_text(text, "doc1", 4, 1).unwrap();
        // Should produce overlapping windows
        assert!(chunks.len() > 1);
        // Each chunk (except the last) should be chunk_size words
        assert_eq!(chunks[0].token_count, 4);
    }

    #[test]
    fn single_chunk_for_tiny_text() {
        let text = "hello world";
        let chunks = chunk_text(text, "doc1", 512, 50).unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn overlap_carried_forward() {
        let text = "a b c d e f";
        // chunk_size=3, overlap=1 → [a b c], [c d e], [e f]
        let chunks = chunk_text(text, "doc1", 3, 1).unwrap();
        assert!(chunks[0].text.contains('c'));
        assert!(chunks[1].text.starts_with('c'));
    }

    #[test]
    fn rejects_invalid_params() {
        assert!(chunk_text("text", "doc", 0, 0).is_err());
        assert!(chunk_text("text", "doc", 3, 3).is_err());
    }
}
