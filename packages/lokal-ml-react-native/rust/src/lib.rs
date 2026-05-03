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
use std::sync::{Mutex, OnceLock};

use lokal_ml_core::{
    engine::{EngineConfig, LokalEngine},
    hardware,
    registry::Registry,
};

// ─── Global engine registry ───────────────────────────────────────────────────
// Maps opaque u32 handles → loaded LokalEngine instances.
// Handles are minted sequentially; 0 is reserved as "invalid".

type EngineStore = Mutex<HashMap<u32, LokalEngine>>;

fn engine_store() -> &'static EngineStore {
    static STORE: OnceLock<EngineStore> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

static NEXT_HANDLE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);

// ─── Embedded registry (stub) ────────────────────────────────────────────────
// In production this is fetched from the CDN on first install.
const EMBEDDED_REGISTRY: &str = include_str!("../../../../registry/models.json");

// ─── Exports ─────────────────────────────────────────────────────────────────

/// Check whether the device meets the minimum hardware requirements for the
/// given model. Returns `true` if capable, `false` if not.
#[no_mangle]
pub unsafe extern "C" fn lokal_check_requirements(model_id: *const c_char) -> bool {
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
    _require_wifi: bool,
    _on_progress: Option<unsafe extern "C" fn(f32)>,
) {
    let _id = unsafe { CStr::from_ptr(model_id).to_string_lossy().to_string() };
    // TODO: Spawn Tokio task that calls lokal_ml_core::downloader::download_model
    // and drives on_progress callback via the JSI event loop.
}

/// Return whether the model file is present in the device cache.
#[no_mangle]
pub unsafe extern "C" fn lokal_is_model_cached(model_id: *const c_char) -> bool {
    let _id = unsafe { CStr::from_ptr(model_id).to_string_lossy() };
    // TODO: check <cache_dir>/<model_id>.gguf existence
    false
}

/// Initialise the inference engine for the given model.
/// Returns an opaque u32 handle (0 on failure).
#[no_mangle]
pub unsafe extern "C" fn lokal_init_engine(model_path: *const c_char) -> u32 {
    let path_str = unsafe { CStr::from_ptr(model_path).to_string_lossy().to_string() };
    let path = std::path::Path::new(&path_str);

    match LokalEngine::load(path, EngineConfig::default()) {
        Ok(engine) => {
            let handle = NEXT_HANDLE.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            engine_store().lock().unwrap().insert(handle, engine);
            handle
        }
        Err(_) => 0,
    }
}

/// Run inference on the engine identified by `handle`, streaming tokens via
/// the provided callback. `on_token` may be NULL (batch-only mode).
#[no_mangle]
pub unsafe extern "C" fn lokal_chat_stream(
    handle: u32,
    prompt: *const c_char,
    on_token: Option<unsafe extern "C" fn(*const c_char)>,
) {
    let prompt_str = unsafe { CStr::from_ptr(prompt).to_string_lossy().to_string() };
    let store = engine_store().lock().unwrap();

    if let Some(engine) = store.get(&handle) {
        let _ = engine.chat_stream(&prompt_str, move |tok| {
            if let Some(cb) = on_token {
                if let Ok(c) = CString::new(tok) {
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
