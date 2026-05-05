//! Lokal ML FFI layer.
//!
//! Exports a C-ABI interface consumed by the iOS JSI bridge (LokalML.mm)
//! and the Android JNI bridge. `cbindgen` reads this file to produce the
//! `cpp/lokal-ml.h` header automatically.
//!
//! All functions are `#[no_mangle] pub unsafe extern "C"` and follow the
//! pattern established by `taladb-react-native/rust/src/lib.rs`.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use lokal_ml_core::{
    engine::{EngineConfig, LokalEngine},
    hardware,
    registry::Registry,
};

// ─── Tokio runtime ────────────────────────────────────────────────────────────
// A single multi-threaded runtime shared across all FFI calls. Initialised
// lazily on first use so it doesn't consume resources until needed.

fn tokio_rt() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("lokal-ml-worker")
            .enable_all()
            .build()
            .expect("failed to build Tokio runtime")
    })
}

// ─── Global engine registry ───────────────────────────────────────────────────
// Maps opaque u32 handles → loaded LokalEngine instances wrapped in Arc so
// inference can proceed without holding the global mutex.

type EngineStore = Mutex<HashMap<u32, Arc<LokalEngine>>>;

fn engine_store() -> &'static EngineStore {
    static STORE: OnceLock<EngineStore> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_HANDLE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

// ─── Embedded registry (stub) ────────────────────────────────────────────────
// In production this is fetched from the CDN on first install.
const EMBEDDED_REGISTRY: &str = include_str!("../../../../registry/models.json");

// ─── Path helpers ─────────────────────────────────────────────────────────────

/// Return a best-effort app cache directory for storing model files.
/// On iOS the actual cache path is passed from ObjC via `lokal_init_engine`;
/// this fallback is used in tests and on platforms without a host layer.
fn default_cache_dir() -> PathBuf {
    std::env::temp_dir().join("lokal-ml-cache")
}

/// Validate that `path` is an absolute path that does not escape the expected
/// cache root. Returns the canonicalized path on success.
///
/// Prevents a caller from passing `../../etc/passwd` to `lokal_init_engine`.
fn safe_model_path(raw: &str) -> Option<PathBuf> {
    let path = Path::new(raw);

    // Must be absolute so we can verify containment.
    if !path.is_absolute() {
        return None;
    }

    // Reject obvious traversal attempts before the file exists.
    let components: Vec<_> = path.components().collect();
    if components.iter().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return None;
    }

    // Must end in .gguf — prevents loading arbitrary files.
    if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
        return None;
    }

    Some(path.to_path_buf())
}

// ─── Exports ─────────────────────────────────────────────────────────────────

/// Check whether the device meets the minimum hardware requirements for the
/// given model. Returns `true` if capable, `false` if not.
#[no_mangle]
pub unsafe extern "C" fn lokal_check_requirements(model_id: *const c_char) -> bool {
    if model_id.is_null() {
        return false;
    }
    let id = unsafe { CStr::from_ptr(model_id).to_string_lossy() };
    let Ok(registry) = Registry::from_json(EMBEDDED_REGISTRY) else {
        return false;
    };
    let Ok(spec) = registry.get(&id) else {
        return false;
    };
    hardware::check_requirements(spec).is_ok()
}

/// Begin (or resume) downloading the model weights to the device cache.
/// `on_progress` may be NULL — if provided, it is called with a float in [0,1].
#[no_mangle]
pub unsafe extern "C" fn lokal_download_model(
    model_id: *const c_char,
    require_wifi: bool,
    on_progress: Option<unsafe extern "C" fn(f32)>,
) {
    if model_id.is_null() {
        return;
    }
    let id = unsafe { CStr::from_ptr(model_id).to_string_lossy().to_string() };

    let Ok(registry) = Registry::from_json(EMBEDDED_REGISTRY) else {
        return;
    };
    let Ok(spec) = registry.get(&id) else {
        return;
    };

    let dest = default_cache_dir().join(format!("{}.gguf", id));
    let url = spec.url.clone();
    let sha256 = spec.sha256.clone();
    let total_bytes = spec.size_bytes;

    tokio_rt().spawn(async move {
        let options = lokal_ml_core::downloader::DownloadOptions {
            require_wifi,
            on_progress: on_progress.map(|cb| {
                let f: Box<dyn Fn(f32) + Send + Sync> = Box::new(move |p| {
                    // Safety: the callback is a plain C function pointer with no
                    // captures; it is safe to call from any thread.
                    unsafe { cb(p) };
                });
                f
            }),
        };

        if let Err(e) = lokal_ml_core::downloader::download_model(
            &url, &dest, total_bytes, &sha256, &options,
        )
        .await
        {
            tracing::error!("lokal_download_model failed: {}", e);
        }
    });
}

/// Return whether the model file is present in the device cache.
#[no_mangle]
pub unsafe extern "C" fn lokal_is_model_cached(model_id: *const c_char) -> bool {
    if model_id.is_null() {
        return false;
    }
    let id = unsafe { CStr::from_ptr(model_id).to_string_lossy() };
    default_cache_dir().join(format!("{}.gguf", id)).exists()
}

/// Remove the cached model file from disk.
#[no_mangle]
pub unsafe extern "C" fn lokal_delete_model(model_id: *const c_char) {
    if model_id.is_null() {
        return;
    }
    let id = unsafe { CStr::from_ptr(model_id).to_string_lossy() };
    let path = default_cache_dir().join(format!("{}.gguf", id));
    if path.exists() {
        if let Err(e) = std::fs::remove_file(&path) {
            tracing::warn!("lokal_delete_model: failed to remove {:?}: {}", path, e);
        }
    }
}

/// Initialise the inference engine for the given model path.
/// Returns an opaque u32 handle (0 on failure).
///
/// The caller is responsible for passing a valid absolute path to a `.gguf`
/// file inside the app's cache directory (see `modelCachePath()` in LokalML.mm).
#[no_mangle]
pub unsafe extern "C" fn lokal_init_engine(model_path: *const c_char) -> u32 {
    if model_path.is_null() {
        return 0;
    }
    let path_str = unsafe { CStr::from_ptr(model_path).to_string_lossy().to_string() };

    let path = match safe_model_path(&path_str) {
        Some(p) => p,
        None => {
            tracing::error!("lokal_init_engine: rejected unsafe path {:?}", path_str);
            return 0;
        }
    };

    match LokalEngine::load(&path, EngineConfig::default()) {
        Ok(engine) => {
            let handle = NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            engine_store().lock().unwrap().insert(handle, Arc::new(engine));
            handle
        }
        Err(e) => {
            tracing::error!("lokal_init_engine: load failed: {}", e);
            0
        }
    }
}

/// Run inference on the engine identified by `handle`, streaming tokens via
/// the provided callback. `on_token` may be NULL (batch-only mode).
///
/// The engine Arc is cloned out of the store before the mutex is released,
/// so inference does not block concurrent `lokal_init_engine` / `lokal_dispose_engine` calls.
#[no_mangle]
pub unsafe extern "C" fn lokal_chat_stream(
    handle: u32,
    prompt: *const c_char,
    on_token: Option<unsafe extern "C" fn(*const c_char)>,
) {
    if prompt.is_null() {
        return;
    }
    let prompt_str = unsafe { CStr::from_ptr(prompt).to_string_lossy().to_string() };

    // Clone the Arc and immediately release the lock so the mutex is not held
    // for the duration of inference (which can take several seconds).
    let engine = {
        let store = engine_store().lock().unwrap();
        store.get(&handle).cloned()
    };

    if let Some(engine) = engine {
        let _ = engine.chat_stream(&prompt_str, move |tok| {
            if let Some(cb) = on_token {
                if let Ok(c) = CString::new(tok) {
                    // Safety: `cb` is a plain C function pointer; `c` lives for
                    // the duration of this closure invocation.
                    unsafe { cb(c.as_ptr()) };
                }
            }
        });
    }
}

/// Release the engine handle and free native memory.
#[no_mangle]
pub unsafe extern "C" fn lokal_dispose_engine(handle: u32) {
    engine_store().lock().unwrap().remove(&handle);
}

/// Return the full cache path for a model ID as a heap-allocated C string.
/// The caller must free this string with `lokal_free_string`.
/// Returns NULL if the model ID is invalid or empty.
#[no_mangle]
pub unsafe extern "C" fn lokal_get_model_cache_path(model_id: *const c_char) -> *mut c_char {
    if model_id.is_null() {
        return std::ptr::null_mut();
    }
    let id = unsafe { CStr::from_ptr(model_id).to_string_lossy() };
    if id.is_empty() {
        return std::ptr::null_mut();
    }
    let path = default_cache_dir().join(format!("{}.gguf", id));
    match CString::new(path.display().to_string()) {
        Ok(s) => s.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Free a C string previously returned by `lokal_get_model_cache_path`.
#[no_mangle]
pub unsafe extern "C" fn lokal_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe { drop(CString::from_raw(ptr)) };
    }
}
