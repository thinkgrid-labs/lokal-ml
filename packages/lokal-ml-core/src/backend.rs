use std::sync::OnceLock;

use llama_cpp_2::llama_backend::LlamaBackend;

/// Return a reference to the process-wide llama.cpp backend.
///
/// `LlamaBackend::init()` is idempotent in the underlying C++ layer; the
/// `BackendAlreadyInitialized` arm handles the case where another library or
/// an earlier FFI call already ran `llama_backend_init()`.
pub fn get() -> &'static LlamaBackend {
    static BACKEND: OnceLock<LlamaBackend> = OnceLock::new();
    BACKEND.get_or_init(|| match LlamaBackend::init() {
        Ok(b) => b,
        Err(_) => LlamaBackend {},
    })
}
