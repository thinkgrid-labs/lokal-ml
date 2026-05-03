/**
 * Shared TypeScript type definitions for @lokal-ml/react-native.
 */

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
   * The model ID to load (must already be downloaded via ModelManager).
   * Example: 'gemma-2b-int4'
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
