import type { LokalPlugin } from '@lokal-ml/react-native';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface TalaPluginOptions {
  /** An open TalaDB database instance. */
  db: unknown; // typed as `TalaDB` once @taladb/react-native types are imported
  /** The collection within TalaDB to store embedded chunks. */
  collection: string;
  /** Base URL of the Lokal ML registry (for embedding model download). */
  registryUrl?: string;
}

export interface IngestOptions {
  /** Documents to ingest. */
  data: { id: string; text: string }[];
  /** Target chunk size in words (default: 512). */
  chunkSize?: number;
  /** Overlap between consecutive chunks in words (default: 50). */
  overlap?: number;
}

// ─── Plugin ───────────────────────────────────────────────────────────────────

/**
 * TalaPlugin
 *
 * Wires the Lokal ML inference engine into a TalaDB local vector store for
 * fully offline RAG. Register via `Lokal.init({ plugins: [new TalaPlugin(...)] })`.
 *
 * @example
 * ```ts
 * import { openDB } from 'taladb';
 * import { TalaPlugin } from '@lokal-ml/taladb-plugin';
 *
 * const db = await openDB('my_app.db');
 * const ai = await Lokal.init({
 *   model: 'gemma-2b-int4',
 *   plugins: [new TalaPlugin({ db, collection: 'knowledge_base' })],
 * });
 *
 * await ai.plugins.TalaRAG.ingest({
 *   data: [{ id: 'doc1', text: 'Enterprise SLAs require a 2-hour response time...' }],
 * });
 * ```
 */
export class TalaPlugin implements LokalPlugin {
  public readonly name = 'TalaRAG';

  private db: unknown;
  private collection: string;

  constructor(options: TalaPluginOptions) {
    this.db = options.db;
    this.collection = options.collection;
  }

  /**
   * Ingest documents into TalaDB.
   *
   * Under the hood this calls the native Rust `lokal_ml_taladb` pipeline:
   * text → chunk → embed (all-MiniLM-L6-v2) → HNSW vector insert.
   *
   * The embedding model is downloaded silently on first call if not cached.
   */
  async ingest(options: IngestOptions): Promise<void> {
    const { data, chunkSize = 512, overlap = 50 } = options;

    // TODO: Bridge call to the native lokal_tala_ingest() FFI function
    // which invokes lokal_ml_taladb::injector::ingest_document() for each doc.
    console.log(
      `[TalaPlugin] Ingesting ${data.length} documents into '${this.collection}' ` +
      `(chunkSize=${chunkSize}, overlap=${overlap})`
    );
  }

  /**
   * Query TalaDB for the top-K chunks most semantically similar to `query`.
   * Used internally by the engine when `useRAG: true` is set on `ai.chat()`.
   *
   * @param query - The user's prompt to search against
   * @param topK  - Number of chunks to retrieve (default: 5)
   * @returns Retrieved text snippets, ordered by similarity score (desc)
   */
  async retrieve(query: string, topK = 5): Promise<string[]> {
    // TODO: embed query → vector search in TalaDB → return text[] 
    console.log(`[TalaPlugin] Retrieving top-${topK} for: "${query}"`);
    return [];
  }
}
