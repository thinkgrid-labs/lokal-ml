//! Resumable model downloader.
//!
//! Downloads large `.gguf` model files (1.5 GB+) to the device's persistent
//! cache directory using HTTP Range requests. Verifies integrity with SHA-256
//! after the download completes.
//!
//! The downloader is designed to survive:
//! - App backgrounding / foreground transitions
//! - Network drops (resumes from the last received byte)
//! - Corrupted partial files (re-verifies from the beginning)


use futures::StreamExt;
use reqwest::Client;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use subtle::ConstantTimeEq;
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tracing::{debug, info, warn};

#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SHA-256 mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("Server does not support range requests — resumption not possible")]
    RangeNotSupported,
}

/// Options for the downloader.

pub struct DownloadOptions {
    /// If `true`, only proceed if the device is connected to Wi-Fi.
    /// Enforcement is the responsibility of the JS/Dart host layer.
    pub require_wifi: bool,
    /// Called with a progress value in `[0.0, 1.0]` as bytes are received.
    pub on_progress: Option<Box<dyn Fn(f32) + Send + Sync>>,
}

impl Default for DownloadOptions {
    fn default() -> Self {
        Self {
            require_wifi: true,
            on_progress: None,
        }
    }
}

/// Download a `.gguf` model file to `dest_path`, resuming from where a
/// previous partial download left off.
///
/// After the download completes the file's SHA-256 digest is compared against
/// `expected_sha256`. If they don't match the partial file is **not** deleted
/// so the caller can decide whether to retry.
pub async fn download_model(
    url: &str,
    dest_path: &Path,
    total_bytes: u64,
    expected_sha256: &str,
    options: &DownloadOptions,
) -> Result<PathBuf, DownloadError> {
    let client = Client::builder()
        .user_agent("lokal-ml/0.1")
        .build()?;

    // Determine how many bytes we already have (partial download resume)
    let already_downloaded = if dest_path.exists() {
        tokio::fs::metadata(dest_path).await?.len()
    } else {
        0
    };

    if already_downloaded == total_bytes {
        info!("Model already fully downloaded at {:?}", dest_path);
        // Still verify hash before returning
        verify_sha256(dest_path, expected_sha256).await?;
        return Ok(dest_path.to_path_buf());
    }

    info!(
        url,
        already_downloaded,
        total_bytes,
        "Starting/resuming model download"
    );

    // Build request with Range header if resuming
    let mut request = client.get(url);
    if already_downloaded > 0 {
        request = request.header("Range", format!("bytes={}-", already_downloaded));
        debug!("Resuming from byte {}", already_downloaded);
    }

    let response = request.send().await?.error_for_status()?;

    // Validate that the server honoured the Range request
    if already_downloaded > 0 && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
        warn!("Server returned 200 instead of 206 — restarting download from scratch");
        // Truncate the partial file and restart
        tokio::fs::remove_file(dest_path).await.ok();
    }

    // Open file in append mode (or create fresh)
    let mut file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dest_path)
        .await?;

    let mut bytes_received = already_downloaded;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        bytes_received += chunk.len() as u64;

        if let Some(cb) = &options.on_progress {
            let progress = bytes_received as f32 / total_bytes as f32;
            cb(progress.clamp(0.0, 1.0));
        }
    }

    file.flush().await?;
    drop(file);

    info!("Download complete — verifying SHA-256");
    verify_sha256(dest_path, expected_sha256).await?;

    info!("Model verified and ready at {:?}", dest_path);
    Ok(dest_path.to_path_buf())
}

/// Compute the SHA-256 digest of a file and compare it to the expected hex string.
async fn verify_sha256(path: &Path, expected: &str) -> Result<(), DownloadError> {
    let mut file = tokio::fs::File::open(path).await?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024]; // 64 KB read buffer

    loop {
        let n = file.read(&mut buf).await?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }

    let actual = hex::encode(hasher.finalize());
    // Constant-time comparison prevents timing side-channels on the hash.
    let equal: bool = actual.as_bytes().ct_eq(expected.as_bytes()).into();
    if !equal {
        return Err(DownloadError::HashMismatch {
            expected: expected.to_string(),
            actual,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn hash_mismatch_returns_error() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.gguf");

        // Write dummy content
        tokio::fs::write(&path, b"not a real model").await.unwrap();

        let result = verify_sha256(&path, "0000000000000000000000000000000000000000000000000000000000000000").await;
        assert!(matches!(result, Err(DownloadError::HashMismatch { .. })));
    }
}
