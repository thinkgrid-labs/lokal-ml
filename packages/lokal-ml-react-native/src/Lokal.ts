import type { LokalConfig, ChatOptions, ChatResponse, PluginRegistry, LokalPlugin } from './types';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const LokalMLNative = (global as any).LokalMLNative as {
  initEngine(config: { model: string; contextSize: number; threads: number }): Promise<number>;
  chat(handle: number, options: Record<string, unknown>): Promise<ChatResponse>;
  disposeEngine(handle: number): void;
};

/**
 * A loaded Lokal ML engine instance.
 *
 * Returned by `Lokal.init()`. Holds a reference to the native engine handle.
 */
export class LokalInstance {
  private _engineHandle: number;
  public readonly plugins: PluginRegistry;

  constructor(engineHandle: number, plugins: LokalPlugin[]) {
    this._engineHandle = engineHandle;
    this.plugins = Object.fromEntries(plugins.map((p) => [p.name, p]));
  }

  /**
   * Execute a prompt and optionally stream tokens to the UI.
   *
   * If `useRAG: true`, the engine automatically searches TalaDB (via the
   * registered TalaPlugin) for relevant context and injects it into the
   * system prompt before generating.
   *
   * @param options - Chat options including prompt, RAG flag, and token callback
   */
  async chat(options: ChatOptions): Promise<ChatResponse> {
    const {
      prompt,
      useRAG = false,
      onToken = null,
      systemPrompt = '',
      maxTokens = 512,
      temperature = 0.7,
    } = options;

    return LokalMLNative.chat(this._engineHandle, {
      prompt,
      useRAG,
      onToken,
      systemPrompt,
      maxTokens,
      temperature,
    });
  }

  /**
   * Release the native engine handle and free model memory.
   * Call this when the engine is no longer needed to avoid memory leaks.
   */
  async dispose(): Promise<void> {
    return LokalMLNative.disposeEngine(this._engineHandle);
  }
}

/**
 * Lokal
 *
 * Entry point for the inference engine. Call `Lokal.init()` once per session
 * after the model has been downloaded via `ModelManager.downloadModel()`.
 */
export const Lokal = {
  /**
   * Initialise the inference engine with the specified model and plugins.
   *
   * Loading the model is a one-time cost (typically 1–3 seconds depending
   * on device). The returned `LokalInstance` should be kept alive for the
   * duration of the session.
   *
   * @param config - Engine configuration (model ID, context size, plugins)
   * @returns A fully initialised `LokalInstance`
   *
   * @example
   * ```ts
   * const ai = await Lokal.init({
   *   model: 'gemma-2b-int4',
   *   plugins: [new TalaPlugin({ db, collection: 'knowledge_base' })],
   * });
   * ```
   */
  init: async (config: LokalConfig): Promise<LokalInstance> => {
    const { model, contextSize = 2048, threads = 4, plugins = [] } = config;

    const engineHandle: number = await LokalMLNative.initEngine({
      model,
      contextSize,
      threads,
    });

    return new LokalInstance(engineHandle, plugins);
  },
};
