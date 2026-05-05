/**
 * @lokal-ml/react-native — TypeScript public API
 *
 * Exposes ModelManager (registry, hardware check, download, cache management)
 * and Lokal.init() (engine initialisation with optional plugin support).
 */

export type {
  ModelSpec,
  ModelTag,
  DownloadOptions,
  ChatOptions,
  ChatResponse,
  LokalConfig,
  LokalPlugin,
  PluginRegistry,
} from './types';

export { ModelManager } from './ModelManager';
export { Lokal, LokalInstance } from './Lokal';

export {
  MODELS,
  getRecommendedModels,
  getModelsByTag,
  getModel,
} from './registry';
