//! # lokal-ml-taladb
//!
//! Optional RAG plugin that wires the Lokal ML inference engine into TalaDB's
//! local vector store.
//!
//! ## Pipeline
//! 1. [`chunker`]  — splits raw text into fixed-size token windows with overlap
//! 2. [`embedder`] — converts chunks to 384-dim vectors via all-MiniLM-L6-v2
//! 3. [`injector`] — writes vectors + text into TalaDB via the HNSW index

pub mod chunker;
pub mod embedder;
pub mod injector;

pub use chunker::{Chunk, ChunkError, chunk_text};
pub use embedder::{Embedder, EmbedError};
pub use injector::{inject_chunks, InjectionError};
