export type OutputType = "answer" | "summary" | "decisions" | "risks" | "actions";

export interface Citation {
  source: string;
  page?: number | null;
  section?: string | null;
  chunk_id: string;
  snippet: string;
  score: number;
}

export interface StructuredItem {
  title: string;
  detail: string;
  citations: Citation[];
}

export interface RetrievedChunk {
  chunk: {
    chunk_id: string;
    text: string;
    metadata: Record<string, unknown>;
  };
  score: number;
  retriever: string;
  rank: number;
}

export interface QueryResponse {
  answer: string;
  output_type: OutputType;
  citations: Citation[];
  items: StructuredItem[];
  confidence: number;
  chunks?: RetrievedChunk[];
  metadata: Record<string, unknown>;
  latency_ms: number;
  provider_used: string;
}

export interface QueryRequest {
  query: string;
  output_type?: OutputType;
  top_k?: number;
  filters?: Record<string, unknown>;
  enable_reranking?: boolean;
  llm_provider?: string;
  include_chunks?: boolean;
}

export interface ProviderInfo {
  available: boolean;
  model?: string;
  error?: string;
}

export interface Health {
  status: string;
  app: string;
  version: string;
  environment: string;
  vector_backend: string;
  vector_count: number;
  bm25_count: number;
  providers: Record<string, ProviderInfo>;
}
