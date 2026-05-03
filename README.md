# 🚀 Lokal ML (@lokal-ml)

> **The Local-First Mobile LLM Infrastructure. Zero friction. Pure Rust. Native Edge AI.**

> [!WARNING]
> **🚧 Active Development — Not Production Ready**  
> Lokal ML is under active construction. APIs are unstable and subject to change. Contributions and early feedback are very welcome — star the repo to follow progress.

---

## 🛑 The Problem

Running Small Language Models (SLMs) like Gemma directly on mobile devices is the future of privacy-first, zero-latency applications. But today, the Developer Experience (DX) is fundamentally broken:

- **The App Store Trap:** Bundling a 1.5GB+ `.gguf` quantized model directly into an app binary destroys user acquisition and violates App Store cellular download limits.
- **The C++ Boilerplate:** React Native and Flutter developers are forced to wrestle with complex C++ wrappers, asynchronous bridging overhead, and memory leaks just to stream tokens.
- **The RAG Fragmentation:** Building offline Retrieval-Augmented Generation (RAG) requires developers to manually stitch together text chunkers, separate embedding models, and local vector databases.

---

## 🌟 The Manifesto

Lokal ML is an open-source, infrastructure-grade SDK designed to bring local, on-device SLMs to React Native and Flutter with absolute zero friction.

Built on a pure-Rust core for maximum memory safety and cross-platform performance, Lokal ML abstracts away the brutal complexities of mobile ML. It handles the hardware constraints, the model delivery, and the vector math — so you can focus on shipping features.

---

## 🏗️ Core Architecture

| Layer | Description |
|---|---|
| 🦀 **Pure Rust Core** | High-performance, memory-safe execution via `llama-cpp-2` (GGUF/Metal/NEON) |
| ⚡ **Zero-Overhead Bridging** | Direct JSI for React Native, `flutter_rust_bridge` FFI for Flutter — instant token streaming, no async bottlenecks |
| 📦 **Shell & Fetch Delivery** | Resumable background downloader that profiles device hardware and fetches model weights post-install. Initial binary stays < 50 MB |
| 🧠 **Plug-and-Play Local RAG** | Optional TalaDB plugin that auto-chunks text, runs a local embedding model, and persists vectors natively — Rust-to-Rust, zero serialisation |

### Repository Structure

```
lokal-ml/
├── packages/
│   ├── lokal-ml-core/                # 🦀 Rust: hardware profiler, downloader, GGUF engine
│   ├── lokal-ml-taladb/              # 🦀 Rust: chunker, embedder, TalaDB vector injector
│   ├── lokal-ml-react-native/        # 📱 React Native JSI bridge + TypeScript API
│   │   └── rust/                     #    C-ABI FFI layer (cbindgen → lokal-ml.h)
│   └── lokal-ml-taladb-plugin/       # 🔌 @lokal-ml/taladb-plugin TypeScript wrapper
└── registry/
    └── models.json                   # Model manifest (URLs, SHA-256, hardware requirements)
```

---

## 💻 Developer Experience (DX)

### Packages

| Package | Status | Description |
|---|---|---|
| `@lokal-ml/react-native` | 🚧 In Development | Core engine — hardware check, model download, GGUF inference |
| `@lokal-ml/taladb-plugin` | 🚧 In Development | Optional RAG layer — offline vector memory via TalaDB |

### The Vision

```ts
import { Lokal, ModelManager } from '@lokal-ml/react-native';
import { TalaPlugin } from '@lokal-ml/taladb-plugin';
import { openDB } from '@taladb/react-native';

// 1. Profiler prevents OOM crashes on older devices
const canRun = await ModelManager.checkRequirements('gemma-2b-int4');
if (!canRun) {
  console.log('Device cannot run local AI — falling back to cloud.');
  return;
}

// 2. Resumable background download (Wi-Fi enforced, fires only if not cached)
await ModelManager.downloadModel('gemma-2b-int4', {
  requireWifi: true,
  onProgress: (p) => setProgress(p),
});

// 3. Connect to local-first storage & initialise engine
const db = await openDB('local_data.db');
const ai = await Lokal.init({
  model: 'gemma-2b-int4',
  plugins: [new TalaPlugin({ db, collection: 'knowledge_base' })],
});

// 4. Ingest your documents (auto-chunked + auto-embedded locally)
await ai.plugins.TalaRAG.ingest({
  data: [{ id: 'policy_1', text: 'Enterprise SLAs require a 2-hour response time...' }],
});

// 5. Stream instantly with embedded RAG context
await ai.chat({
  prompt: 'What is the enterprise SLA response time?',
  useRAG: true,
  onToken: (token) => process.stdout.write(token),
});
```

---

## App Store Compliance

- ✅ Initial app binary < 50 MB — no model weights bundled
- ✅ Weights fetched post-install via HTTP Range (resumable, survives backgrounding)
- ✅ `requireWifi: true` enforced by default
- ✅ Files stored in OS-designated app data directory (excluded from iCloud backup)

---

## Development

```bash
# Clone
git clone https://github.com/thinkgrid-labs/lokal-ml
cd lokal-ml

# Rust workspace (requires cmake for llama.cpp)
cargo check
cargo test

# JS packages
pnpm install
pnpm typecheck
```

> **Prerequisites:** Rust stable, cmake, Node ≥ 18, pnpm ≥ 9

---

## License

MIT — © 2026 thinkgrid-labs
