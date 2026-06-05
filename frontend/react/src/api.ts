import type { Health, QueryRequest, QueryResponse } from "./types";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const getHealth = () => request<Health>("/health");

export const runQuery = (req: QueryRequest) =>
  request<QueryResponse>("/api/v1/query", {
    method: "POST",
    body: JSON.stringify(req),
  });

export const ingest = (rebuild = false) =>
  request<{ documents_processed: number; chunks_created: number }>(
    "/api/v1/ingest",
    {
      method: "POST",
      body: JSON.stringify({ paths: [], rebuild_index: rebuild }),
    },
  );
