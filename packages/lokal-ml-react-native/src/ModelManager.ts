import type { DownloadOptions, ModelSpec } from './types';
import { MODELS, getRecommendedModels, getModelsByTag, getModel } from './registry';

// The JSI HostObject is installed on the JS global via LokalML.mm — not on
// NativeModules, which is a separate React Native registry.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const LokalMLNative = (global as any).LokalMLNative as {
  checkRequirements(modelId: string): boolean;
  downloadModel(
    modelId: string,
    requireWifi: boolean,
    onProgress: ((p: number) => void) | null
  ): Promise<void>;
  isModelCached(modelId: string): boolean;
  deleteModel(modelId: string): void;
};

/**
 * ModelManager
 *
 * Handles hardware validation, model lifecycle (download, cache, delete),
 * and registry access. All native methods are async and resolve/reject on
 * the JS thread via the JSI bridge.
 */
export const ModelManager = {
  // ── Registry ───────────────────────────────────────────────────────────────

  /**
   * Return all models in the built-in registry.
   *
   * No native call required — the registry is bundled with the JS package.
   */
  listModels(): ModelSpec[] {
    return Object.values(MODELS);
  },

  /**
   * Return models marked as top recommendations (currently the Gemma 4 E series).
   */
  listRecommended(): ModelSpec[] {
    return getRecommendedModels();
  },

  /**
   * Return models that carry a specific tag.
   *
   * @example
   * ModelManager.listByTag('tier:nano')    // all nano-tier models
   * ModelManager.listByTag('medical')      // MedGemma and similar
   * ModelManager.listByTag('recommended')  // same as listRecommended()
   */
  listByTag(tag: ModelSpec['tags'][number]): ModelSpec[] {
    return getModelsByTag(tag);
  },

  /**
   * Look up a model spec by its stable ID.
   * Returns `undefined` for unknown IDs (e.g. custom GGUF paths).
   */
  getModel(id: string): ModelSpec | undefined {
    return getModel(id);
  },

  // ── Device capability ──────────────────────────────────────────────────────

  /**
   * Evaluate whether the current device meets the minimum hardware requirements
   * for the specified model.
   *
   * Checks available RAM, CPU architecture, and OS version constraints.
   * Call this before `downloadModel` to gracefully degrade on older devices.
   *
   * @param modelId - Registry model ID (e.g. 'gemma4-e2b')
   * @returns `true` if the device is capable, `false` otherwise.
   */
  async checkRequirements(modelId: string): Promise<boolean> {
    return LokalMLNative.checkRequirements(modelId);
  },

  // ── Model lifecycle ────────────────────────────────────────────────────────

  /**
   * Download a model's weights to the device's persistent cache.
   *
   * Uses HTTP Range requests — safe to call multiple times; resumes from
   * where a previous attempt left off. The download runs on a Tokio background
   * task and does not block the UI.
   *
   * @param modelId - Registry model ID
   * @param options - Optional: wifi requirement, progress callback
   */
  async downloadModel(modelId: string, options: DownloadOptions = {}): Promise<void> {
    const { requireWifi = true, onProgress } = options;
    return LokalMLNative.downloadModel(modelId, requireWifi, onProgress ?? null);
  },

  /**
   * Check whether a model's weights are already present on-device.
   *
   * @param modelId - Registry model ID
   */
  async isModelCached(modelId: string): Promise<boolean> {
    return LokalMLNative.isModelCached(modelId);
  },

  /**
   * Remove a model's weight file from device storage to reclaim space.
   *
   * @param modelId - Registry model ID
   */
  async deleteModel(modelId: string): Promise<void> {
    return LokalMLNative.deleteModel(modelId);
  },
};
