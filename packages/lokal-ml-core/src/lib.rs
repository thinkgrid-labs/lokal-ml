//! # lokal-ml-core
//!
//! Rust inference core for the Lokal ML SDK.
//!
//! ## Modules
//! - [`hardware`]   — Device RAM/arch profiler (runs before any model download)
//! - [`downloader`] — Resumable, SHA-256 verified model fetcher
//! - [`engine`]     — GGUF inference engine via llama.cpp
//! - [`streaming`]  — Token stream producer
//! - [`registry`]   — Model registry manifest parser

pub mod downloader;
pub mod engine;
pub mod hardware;
pub mod registry;
pub mod streaming;

// Re-export top-level error type
pub use engine::EngineError;
pub use hardware::HardwareError;
pub use downloader::DownloadError;
