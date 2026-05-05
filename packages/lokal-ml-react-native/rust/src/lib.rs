//! Lokal ML FFI layer.
//!
//! Exports a C-ABI interface consumed by the iOS JSI bridge (LokalML.mm)
//! and the Android JNI bridge. `cbindgen` reads this file to produce the
//! `cpp/lokal-ml.h` header automatically.
//!
//! All functions are `#[no_mangle] pub unsafe extern "C"` and follow the
//! pattern established by `taladb-react-native/rust/src/lib.rs`.
//!
//! # `user_data` convention
//! Callbacks that may fire from a background thread carry a `*mut c_void
//! user_data` parameter. The caller (ObjC/Kotlin) casts a heap-allocated
//! context struct to `void*`, passes it here, and reconstructs the pointer
//! inside the callback. On the Rust side the pointer is held as `usize` while
//! inside a `Send` closure to satisfy the trait bound without introducing UB.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use lokal_ml_core::{
    engine::{EngineConfig, LokalEngine},
    hardware,
    registry::Registry,
};

// ─── Tokio runtime ────────────────────────────────────────────────────────────

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

type EngineStore = Mutex<HashMap<u32, Arc<LokalEngine>>>;

fn engine_store() -> &'static EngineStore {
    static STORE: OnceLock<EngineStore> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_HANDLE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

// ─── Embedded registry ────────────────────────────────────────────────────────

const EMBEDDED_REGISTRY: &str = include_str!("../../../../registry/models.json");

// ─── Path helpers ─────────────────────────────────────────────────────────────

fn default_cache_dir() -> PathBuf {
    std::env::temp_dir().join("lokal-ml-cache")
}

fn safe_model_path(raw: &str) -> Option<PathBuf> {
    let path = Path::new(raw);
    if !path.is_absolute() {
        return None;
    }
    if path.components().any(|c| matches!(c, std::path::Component::ParentDir)) {
        return None;
    }
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
///
/// `on_progress(progress, user_data)` fires with a float in \[0, 1\]; may be NULL.
/// `on_complete(success, user_data)` fires when the download finishes or fails;
/// the caller must free any heap-allocated `user_data` inside this callback.
#[no_mangle]
pub unsafe extern "C" fn lokal_download_model(
    model_id: *const c_char,
    require_wifi: bool,
    on_progress: Option<unsafe extern "C" fn(f32, *mut c_void)>,
    on_progress_user_data: *mut c_void,
    on_complete: Option<unsafe extern "C" fn(bool, *mut c_void)>,
    on_complete_user_data: *mut c_void,
) {
    if model_id.is_null() {
        if let Some(cb) = on_complete {
            unsafe { cb(false, on_complete_user_data) };
        }
        return;
    }
    let id = unsafe { CStr::from_ptr(model_id).to_string_lossy().to_string() };

    let Ok(registry) = Registry::from_json(EMBEDDED_REGISTRY) else {
        if let Some(cb) = on_complete {
            unsafe { cb(false, on_complete_user_data) };
        }
        return;
    };
    let Ok(spec) = registry.get(&id) else {
        if let Some(cb) = on_complete {
            unsafe { cb(false, on_complete_user_data) };
        }
        return;
    };

    let dest = default_cache_dir().join(format!("{}.gguf", id));
    let url = spec.url.clone();
    let sha256 = spec.sha256.clone();
    let total_bytes = spec.size_bytes;

    // Cast raw pointers to usize for Send-compatibility inside async closures.
    let on_progress_ud = on_progress_user_data as usize;
    let on_complete_ud = on_complete_user_data as usize;

    tokio_rt().spawn(async move {
        let options = lokal_ml_core::downloader::DownloadOptions {
            require_wifi,
            on_progress: on_progress.map(|cb| {
                let f: Box<dyn Fn(f32) + Send + Sync> = Box::new(move |p| {
                    unsafe { cb(p, on_progress_ud as *mut c_void) };
                });
                f
            }),
        };

        let success = match lokal_ml_core::downloader::download_model(
            &url,
            &dest,
            total_bytes,
            &sha256,
            &options,
        )
        .await
        {
            Ok(_) => true,
            Err(e) => {
                tracing::error!("lokal_download_model failed: {}", e);
                false
            }
        };

        if let Some(cb) = on_complete {
            unsafe { cb(success, on_complete_ud as *mut c_void) };
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
/// `on_token`. Blocks until inference finishes, then calls `on_complete`.
///
/// Both callbacks may be NULL. The `user_data` pointers are passed through
/// opaquely — the caller is responsible for allocation and freeing (typically
/// inside `on_complete`).
///
/// This function is synchronous; call it from a background thread (e.g. a GCD
/// concurrent queue or a Kotlin coroutine dispatcher).
#[no_mangle]
pub unsafe extern "C" fn lokal_chat_stream(
    handle: u32,
    prompt: *const c_char,
    on_token: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    on_token_user_data: *mut c_void,
    on_complete: Option<unsafe extern "C" fn(u32, u64, *mut c_void)>,
    on_complete_user_data: *mut c_void,
) {
    if prompt.is_null() {
        if let Some(cb) = on_complete {
            unsafe { cb(0, 0, on_complete_user_data) };
        }
        return;
    }
    let prompt_str = unsafe { CStr::from_ptr(prompt).to_string_lossy().to_string() };

    let engine = {
        let store = engine_store().lock().unwrap();
        store.get(&handle).cloned()
    };

    let Some(engine) = engine else {
        tracing::warn!("lokal_chat_stream: invalid handle {}", handle);
        if let Some(cb) = on_complete {
            unsafe { cb(0, 0, on_complete_user_data) };
        }
        return;
    };

    // Cast to usize so the closure satisfies `Send` without UB: the pointer is
    // only reconstructed back to `*mut c_void` immediately before the C call,
    // which happens on the same thread that owns the context.
    let on_token_ud = on_token_user_data as usize;

    let generated = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let gen_for_closure = Arc::clone(&generated);

    let start = std::time::Instant::now();

    let _ = engine.chat_stream(&prompt_str, move |tok| {
        gen_for_closure.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Some(cb) = on_token {
            if let Ok(c) = CString::new(tok) {
                unsafe { cb(c.as_ptr(), on_token_ud as *mut c_void) };
            }
        }
    });

    let elapsed_ms = start.elapsed().as_millis() as u64;
    let n_generated = generated.load(std::sync::atomic::Ordering::Relaxed);

    if let Some(cb) = on_complete {
        unsafe { cb(n_generated, elapsed_ms, on_complete_user_data) };
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
