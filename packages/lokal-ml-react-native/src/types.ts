/**
 * Shared TypeScript type definitions for @lokal-ml/react-native.
 */

// ─── Model Registry ───────────────────────────────────────────────────────────

/**
 * Capability and tier tags attached to each model spec.
 *
 * Tier meanings:
 *   tier:nano       — < 1 GB download, < 1.5 GB RAM — any modern phone
 *   tier:compact    — 1–4 GB download, 2–4 GB RAM  — mid-range+
 *   tier:edge-plus  — 4–6 GB download, 4–8 GB RAM  — flagship only
 */
export type ModelTag =
  | 'chat'
  | 'embedding'
  | 'multimodal'
  | 'medical'
  | 'edge'
  | 'tier:nano'
  | 'tier:compact'
  | 'tier:edge-plus'
  | 'recommended'
  | 'requires-hf-token';

/** A single entry in the Lokal ML model registry. */
export interface ModelSpec {
  /** Stable model ID (e.g. 'gemma4-e2b') */
  id: string;
  /** HuggingFace GGUF download URL */
  url: string;
  /** SHA-256 hex digest for integrity check */
  sha256: string;
  /** Download size in bytes */
  size_bytes: number;
  /** Minimum device RAM in MB */
  min_ram_mb: number;
  /** Human-readable description */
  description: string;
  /** Capability and tier tags */
  tags: ModelTag[];
  /** True if this is a highlighted top recommendation */
  recommended: boolean;
}

// ─── Download ────────────────────────────────────────────────────────────────

/** Options for ModelManager.downloadModel() */
export interface DownloadOptions {
  /**
   * If true, the download will only proceed when the device is on Wi-Fi.
   * Recommended: true (default) for App Store cellular download compliance.
   */
  requireWifi?: boolean;

  /**
   * Progress callback. Called with a value in [0, 1] as bytes are received.
   */
  onProgress?: (percent: number) => void;
}

// ─── Chat ─────────────────────────────────────────────────────────────────────

/** Options for LokalInstance.chat() */
export interface ChatOptions {
  /** The user's prompt. */
  prompt: string;

  /**
   * If true, the engine will search TalaDB for relevant context and inject
   * it into the LLM system prompt before generating. Requires the TalaPlugin
   * to be registered via Lokal.init({ plugins }).
   */
  useRAG?: boolean;

  /**
   * Token streaming callback. Called on each generated token, dispatched to
   * the JS thread via JSI — zero async bridge overhead.
   */
  onToken?: (token: string) => void;

  /** System prompt to prepend to every conversation turn. */
  systemPrompt?: string;

  /** Maximum number of tokens to generate (default: 512). */
  maxTokens?: number;

  /** Sampling temperature in [0, 1] (default: 0.7). */
  temperature?: number;
}

/** The result returned after a full chat turn completes. */
export interface ChatResponse {
  /** The full generated text (concatenation of all streamed tokens). */
  text: string;
  /** Number of prompt tokens consumed. */
  promptTokens: number;
  /** Number of tokens generated. */
  generatedTokens: number;
  /** Total inference time in milliseconds. */
  inferenceMs: number;
}

// ─── Engine Config ─────────────────────────────────────────────────────────

/** Plugin interface — implement this to extend the engine. */
export interface LokalPlugin {
  /** Plugin display name, used as the key in LokalInstance.plugins. */
  readonly name: string;
}

/** Configuration passed to Lokal.init() */
export interface LokalConfig {
  /**
   * The model to load. Two forms are accepted:
   *
   *   1. A registry model ID (e.g. `'gemma4-e2b'`) — the weights must
   *      already be downloaded via `ModelManager.downloadModel()`.
   *
   *   2. An absolute path to a local `.gguf` file you manage yourself
   *      (e.g. `'/var/mobile/Containers/.../custom.gguf'`). This lets
   *      you ship your own quantized model outside the built-in registry.
   */
  model: string;

  /**
   * Context window size in tokens (default: 2048).
   */
  contextSize?: number;

  /**
   * Number of CPU threads to use (default: 4).
   */
  threads?: number;

  /**
   * Registered plugins (e.g. TalaPlugin for RAG).
   */
  plugins?: LokalPlugin[];
}

// ─── Plugin Registry ───────────────────────────────────────────────────────

/** Dynamic plugin registry, keyed by plugin name. */
export type PluginRegistry = Record<string, LokalPlugin>;
