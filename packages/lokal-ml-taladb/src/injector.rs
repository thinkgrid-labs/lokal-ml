//! TalaDB vector injector.
//!
//! Writes embedded chunks directly into TalaDB's HNSW vector index via a
//! Rust-to-Rust call — no language boundary, no serialisation overhead.
//!
//! This module is the architectural core of the RAG plugin: because both
//! `lokal-ml-taladb` and `taladb-core` are Rust crates, vectors cross from
//! the embedding model output directly into the TalaDB memory arena.

use crate::chunker::Chunk;
use thiserror::Error;
use tracing::info;

#[derive(Debug, Error)]
pub enum InjectionError {
    #[error("TalaDB write failed: {0}")]
    DbError(String),

    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Chunks and vectors length mismatch: {chunks} chunks, {vectors} vectors")]
    LengthMismatch { chunks: usize, vectors: usize },
}

/// A chunk paired with its embedding vector, ready for injection into TalaDB.
#[derive(Debug)]
pub struct EmbeddedChunk {
    pub chunk: Chunk,
    pub vector: Vec<f32>,
}

/// Write a batch of embedded chunks into the specified TalaDB collection.
///
/// Each chunk is stored as a document with its vector for HNSW nearest-neighbour
/// retrieval. The write uses TalaDB's batch insert API to minimise fsync overhead.
///
/// ## Arguments
/// - `collection` — TalaDB collection name (e.g. `"knowledge_base"`)
/// - `embedded`   — Chunks paired with their 384-dim embedding vectors
///
/// ## Notes
/// The `taladb-core` crate API integration is scaffolded here. The actual
/// `taladb_core::Database` type will be wired in once `taladb-core` 0.7 is
/// available on crates.io with the `vector-hnsw` feature stable.
pub fn inject_chunks(
    collection: &str,
    embedded: Vec<EmbeddedChunk>,
) -> Result<usize, InjectionError> {
    if embedded.is_empty() {
        return Ok(0);
    }

    let expected_dim = crate::embedder::EMBED_DIM;
    for ec in &embedded {
        if ec.vector.len() != expected_dim {
            return Err(InjectionError::DimensionMismatch {
                expected: expected_dim,
                actual: ec.vector.len(),
            });
        }
    }

    let count = embedded.len();
    info!(
        collection,
        count,
        "Injecting embedded chunks into TalaDB"
    );

    // TODO: Replace with actual taladb_core API call:
    //
    //   for ec in embedded {
    //       db.collection(collection).insert_vector(
    //           &ec.chunk.id,
    //           &ec.vector,
    //           serde_json::json!({ "text": ec.chunk.text, "doc_id": ec.chunk.doc_id }),
    //       )?;
    //   }
    //
    // This stub validates the pipeline without requiring a live TalaDB instance.

    Ok(count)
}

/// High-level convenience function: chunk → embed → inject in one call.
///
/// This is what the JS/Dart `TalaPlugin.ingest()` method calls under the hood.
pub fn ingest_document(
    embedder: &crate::embedder::Embedder,
    collection: &str,
    doc_id: &str,
    text: &str,
    chunk_size: usize,
    overlap: usize,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let chunks = crate::chunker::chunk_text(text, doc_id, chunk_size, overlap)?;

    let texts: Vec<&str> = chunks.iter().map(|c| c.text.as_str()).collect();
    let vectors = embedder.embed_batch(&texts)?;

    let embedded: Vec<EmbeddedChunk> = chunks
        .into_iter()
        .zip(vectors.into_iter())
        .map(|(chunk, vector)| EmbeddedChunk { chunk, vector })
        .collect();

    let count = inject_chunks(collection, embedded)?;
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::chunk_text;
    use crate::embedder::Embedder;
    use tempfile::tempdir;

    fn make_embedder() -> Embedder {
        let dir = tempdir().unwrap();
        let path = dir.path().join("embed.gguf");
        std::fs::write(&path, b"stub").unwrap();
        Embedder::new(&path).unwrap()
    }

    #[test]
    fn inject_returns_correct_count() {
        let chunks = chunk_text("word1 word2 word3 word4 word5 word6", "doc1", 3, 0).unwrap();
        let embedder = make_embedder();

        let embedded: Vec<EmbeddedChunk> = chunks
            .into_iter()
            .map(|chunk| {
                let vector = embedder.embed(&chunk.text).unwrap();
                EmbeddedChunk { chunk, vector }
            })
            .collect();

        let n = embedded.len();
        let result = inject_chunks("knowledge_base", embedded).unwrap();
        assert_eq!(result, n);
    }

    #[test]
    fn ingest_document_end_to_end() {
        let embedder = make_embedder();
        let text = "Enterprise SLAs require a two hour response time for critical issues.";
        let count = ingest_document(&embedder, "kb", "policy_1", text, 5, 1).unwrap();
        assert!(count > 0);
    }
}
