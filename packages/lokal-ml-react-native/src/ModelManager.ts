import type { DownloadOptions } from './types';

// The JSI HostObject is registered on the JS global (not NativeModules) via
// jsi::Runtime::global().setProperty() in LokalML.mm / LokalML.cpp.
// NativeModules is a separate React Native registry and will not contain it.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const LokalMLNative = (global as any).LokalMLNative as {
  checkRequirements(modelId: string): boolean;
  downloadModel(modelId: string, requireWifi: boolean, onProgress: ((p: number) => void) | null): void;
  isModelCached(modelId: string): boolean;
  deleteModel(modelId: string): void;
};

/**
 * ModelManager
 *
 * Handles hardware validation and model lifecycle (download, cache, delete).
 * All methods are async and resolve/reject on the JS thread.
 */
export const ModelManager = {
  /**
   * Evaluate whether the current device meets the minimum hardware requirements
   * for the specified model.
   *
   * Checks available RAM, CPU architecture, and OS version constraints.
   * Call this before `downloadModel` to gracefully degrade on older devices.
   *
   * @param modelId - Stable model identifier (e.g. 'gemma-2b-int4')
   * @returns `true` if the device is capable, `false` otherwise.
   */
  checkRequirements: async (modelId: string): Promise<boolean> => {
    return LokalMLNative.checkRequirements(modelId);
  },

  /**
   * Download a model's weights to the device's persistent cache.
   *
   * Uses HTTP Range requests — safe to call multiple times; resumes from
   * where a previous attempt left off. The download runs on a background
   * thread and does not block the UI.
   *
   * @param modelId - Stable model identifier
   * @param options - Optional download configuration (wifi requirement, progress)
   */
  downloadModel: async (
    modelId: string,
    options: DownloadOptions = {}
  ): Promise<void> => {
    const { requireWifi = true, onProgress } = options;
    return LokalMLNative.downloadModel(modelId, requireWifi, onProgress ?? null);
  },

  /**
   * Check whether a model's weights are already present on-device.
   *
   * @param modelId - Stable model identifier
   */
  isModelCached: async (modelId: string): Promise<boolean> => {
    return LokalMLNative.isModelCached(modelId);
  },

  /**
   * Remove a model's weight file from device storage to reclaim space.
   *
   * @param modelId - Stable model identifier
   */
  deleteModel: async (modelId: string): Promise<void> => {
    return LokalMLNative.deleteModel(modelId);
  },
};
