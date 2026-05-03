import { NativeModules } from 'react-native';
import type { DownloadOptions } from './types';

// Access the JSI HostObject injected by the native layer.
// On iOS: registered in LokalML.mm via jsi::Runtime::global().setProperty()
// On Android: registered in LokalML.cpp via the same JSI global.
const { LokalMLNative } = NativeModules;

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
