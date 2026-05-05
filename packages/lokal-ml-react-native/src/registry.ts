/**
 * Typed model registry for @lokal-ml/react-native.
 *
 * This mirrors registry/models.json (the Rust source of truth) as a
 * strongly-typed TypeScript module so JS consumers can enumerate and
 * filter models at runtime without any native bridge call.
 *
 * Usage:
 *   import { MODELS, getRecommendedModels } from '@lokal-ml/react-native';
 *   const picks = getRecommendedModels();         // top picks
 *   const nano  = getModelsByTag('tier:nano');    // all nano-tier models
 */

import type { ModelSpec } from './types';

export const MODELS: Record<string, ModelSpec> = {

  // ── Gemma 4 — Recommended (edge-optimised, multimodal, 128K ctx) ──────────

  'gemma4-e2b': {
    id: 'gemma4-e2b',
    url: 'https://huggingface.co/bartowski/gemma-4-e2b-it-GGUF/resolve/main/gemma-4-e2b-it-Q4_K_M.gguf',
    sha256: 'PLACEHOLDER_SHA256_GEMMA4_E2B',
    size_bytes: 3_200_000_000,
    min_ram_mb: 5000,
    description:
      'Gemma 4 E2B (2.3B active / 5.1B total with embeddings) — ' +
      "Google's edge-optimised multimodal model. 128K context. " +
      'Best overall choice for mobile.',
    tags: ['chat', 'multimodal', 'edge', 'tier:edge-plus', 'recommended', 'requires-hf-token'],
    recommended: true,
  },

  'gemma4-e4b': {
    id: 'gemma4-e4b',
    url: 'https://huggingface.co/bartowski/gemma-4-e4b-it-GGUF/resolve/main/gemma-4-e4b-it-Q4_K_M.gguf',
    sha256: 'PLACEHOLDER_SHA256_GEMMA4_E4B',
    size_bytes: 5_000_000_000,
    min_ram_mb: 7500,
    description:
      'Gemma 4 E4B (4.5B active / 8B total with embeddings) — ' +
      'highest-quality edge model for flagship phones. 128K context, multimodal.',
    tags: ['chat', 'multimodal', 'edge', 'tier:edge-plus', 'recommended', 'requires-hf-token'],
    recommended: true,
  },

  // ── Gemma 3 — Compact (official Google QAT Q4_0) ─────────────────────────

  'gemma3-4b': {
    id: 'gemma3-4b',
    url: 'https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/resolve/main/gemma-3-4b-it-q4_0.gguf',
    sha256: 'PLACEHOLDER_SHA256_GEMMA3_4B',
    size_bytes: 2_520_000_000,
    min_ram_mb: 4000,
    description:
      'Gemma 3 4B — official Google QAT Q4_0 quantization. ' +
      'Solid mid-range choice, excellent instruction following. 128K context.',
    tags: ['chat', 'multimodal', 'tier:compact', 'requires-hf-token'],
    recommended: false,
  },

  'gemma3-1b': {
    id: 'gemma3-1b',
    url: 'https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf/resolve/main/gemma-3-1b-it-q4_0.gguf',
    sha256: 'PLACEHOLDER_SHA256_GEMMA3_1B',
    size_bytes: 620_000_000,
    min_ram_mb: 1024,
    description:
      'Gemma 3 1B — official Google QAT Q4_0. Ultra-light nano model, ' +
      'runs on any modern phone. Best for low-latency assistants.',
    tags: ['chat', 'tier:nano', 'requires-hf-token'],
    recommended: false,
  },

  // ── MedGemma — Medical specialisation ────────────────────────────────────

  'medgemma-4b': {
    id: 'medgemma-4b',
    url: 'https://huggingface.co/bartowski/medgemma-4b-it-GGUF/resolve/main/medgemma-4b-it-Q4_K_M.gguf',
    sha256: 'PLACEHOLDER_SHA256_MEDGEMMA_4B',
    size_bytes: 2_520_000_000,
    min_ram_mb: 4000,
    description:
      'MedGemma 4B — Gemma 3 fine-tuned by Google for medical text and ' +
      'image comprehension. For health and wellness apps. 128K context.',
    tags: ['chat', 'multimodal', 'medical', 'tier:compact', 'requires-hf-token'],
    recommended: false,
  },

  // ── Qwen3 — Alternative (open, strong reasoning, 256K ctx) ───────────────

  'qwen3-4b': {
    id: 'qwen3-4b',
    url: 'https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf',
    sha256: 'PLACEHOLDER_SHA256_QWEN3_4B',
    size_bytes: 2_520_000_000,
    min_ram_mb: 3500,
    description:
      'Qwen3 4B — reasoning performance rivals 72B-class models. ' +
      '256K context. Great Gemma alternative on mid-range devices.',
    tags: ['chat', 'tier:compact'],
    recommended: false,
  },

  'qwen3-1.7b': {
    id: 'qwen3-1.7b',
    url: 'https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf',
    sha256: 'PLACEHOLDER_SHA256_QWEN3_1_7B',
    size_bytes: 1_070_000_000,
    min_ram_mb: 1800,
    description:
      'Qwen3 1.7B — strong reasoning in 1 GB. 256K context. ' +
      'Good when Gemma 3 1B capability is insufficient.',
    tags: ['chat', 'tier:nano'],
    recommended: false,
  },

  'qwen3-0.6b': {
    id: 'qwen3-0.6b',
    url: 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_K_M.gguf',
    sha256: 'PLACEHOLDER_SHA256_QWEN3_0_6B',
    size_bytes: 390_000_000,
    min_ram_mb: 768,
    description:
      'Qwen3 0.6B — smallest practical chat model (~390 MB). ' +
      'Runs on entry-level phones. Ideal for simple assistants.',
    tags: ['chat', 'tier:nano'],
    recommended: false,
  },

  // ── Embedding (RAG only) ──────────────────────────────────────────────────

  'all-minilm-l6-v2': {
    id: 'all-minilm-l6-v2',
    url: 'https://huggingface.co/second-state/All-MiniLM-L6-v2-Embedding-GGUF/resolve/main/all-MiniLM-L6-v2-Q8_0.gguf',
    sha256: 'PLACEHOLDER_SHA256_MINILM',
    size_bytes: 23_068_672,
    min_ram_mb: 256,
    description:
      'all-MiniLM-L6-v2 embedding model, 8-bit — 384-dim vectors. ' +
      'Required by TalaDB RAG. Not a chat model.',
    tags: ['embedding'],
    recommended: false,
  },
};

// ─── Convenience helpers ──────────────────────────────────────────────────────

/** Return all models marked as recommended. */
export function getRecommendedModels(): ModelSpec[] {
  return Object.values(MODELS).filter((m) => m.recommended);
}

/** Return all models that carry a specific tag. */
export function getModelsByTag(tag: ModelSpec['tags'][number]): ModelSpec[] {
  return Object.values(MODELS).filter((m) => m.tags.includes(tag));
}

/** Look up a single model by ID. Returns undefined if not found. */
export function getModel(id: string): ModelSpec | undefined {
  return MODELS[id];
}
