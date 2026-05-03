//! Hardware profiler.
//!
//! Runs before any model download or load attempt to prevent OOM crashes on
//! devices that don't meet the minimum requirements for a given model.

use sysinfo::System;
use thiserror::Error;
use tracing::info;

use crate::registry::ModelSpec;

/// Errors returned by the hardware profiler.
#[derive(Debug, Error)]
pub enum HardwareError {
    #[error("Insufficient RAM: device has {available_mb}MB available, model requires {required_mb}MB")]
    InsufficientRam {
        available_mb: u64,
        required_mb: u64,
    },

    #[error("Unsupported architecture: {0}")]
    UnsupportedArch(String),

    #[error("OS version too old: device is {device}, minimum is {minimum}")]
    OsVersionTooOld { device: String, minimum: String },
}

/// A snapshot of the device's hardware profile.
#[derive(Debug, Clone)]
pub struct DeviceProfile {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub arch: String,
    pub os_name: String,
}

impl DeviceProfile {
    /// Collect the current device profile using `sysinfo`.
    pub fn collect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let total_ram_mb = sys.total_memory() / 1024 / 1024;
        let available_ram_mb = sys.available_memory() / 1024 / 1024;
        let arch = std::env::consts::ARCH.to_string();
        let os_name = System::name().unwrap_or_else(|| "Unknown".to_string());

        info!(
            total_ram_mb,
            available_ram_mb,
            arch = %arch,
            os = %os_name,
            "Device profile collected"
        );

        DeviceProfile {
            total_ram_mb,
            available_ram_mb,
            arch,
            os_name,
        }
    }
}

/// Check whether the current device meets the requirements for a given model spec.
///
/// Returns `Ok(())` if the device is capable, or a typed [`HardwareError`] describing
/// exactly why the device falls short.
pub fn check_requirements(spec: &ModelSpec) -> Result<(), HardwareError> {
    let profile = DeviceProfile::collect();

    // Enforce minimum available RAM — use available rather than total to account for
    // OS overhead already eating into memory at runtime.
    if profile.available_ram_mb < spec.min_ram_mb {
        return Err(HardwareError::InsufficientRam {
            available_mb: profile.available_ram_mb,
            required_mb: spec.min_ram_mb,
        });
    }

    // Reject unsupported architectures
    match profile.arch.as_str() {
        "aarch64" | "x86_64" => {}
        other => {
            return Err(HardwareError::UnsupportedArch(other.to_string()));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ModelSpec;

    fn make_spec(min_ram_mb: u64) -> ModelSpec {
        ModelSpec {
            id: "test-model".to_string(),
            url: "https://example.com/model.gguf".to_string(),
            sha256: "abc123".to_string(),
            size_bytes: 0,
            min_ram_mb,
        }
    }

    #[test]
    fn passes_for_low_requirement() {
        // Any real device should have at least 128 MB free
        let spec = make_spec(128);
        assert!(check_requirements(&spec).is_ok());
    }

    #[test]
    fn fails_for_impossible_requirement() {
        // No consumer device has 1 TB of RAM
        let spec = make_spec(1_000_000_000);
        assert!(matches!(
            check_requirements(&spec),
            Err(HardwareError::InsufficientRam { .. })
        ));
    }
}
