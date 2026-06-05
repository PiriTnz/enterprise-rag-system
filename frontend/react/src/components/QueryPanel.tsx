import { useState } from "react";
import { runQuery } from "../api";
import type { OutputType, QueryResponse } from "../types";

interface Props {
  onResult: (r: QueryResponse) => void;
  loading: boolean;
  setLoading: (b: boolean) => void;
}

export function QueryPanel({ onResult, loading, setLoading }: Props) {
  const [query, setQuery] = useState("");
  const [outputType, setOutputType] = useState<OutputType>("answer");
  const [topK, setTopK] = useState(5);
  const [provider, setProvider] = useState("auto");
  const [rerank, setRerank] = useState(true);

  const submit = async () => {
    if (!query.trim()) return;
    setLoading(true);
    try {
      const resp = await runQuery({
        query,
        output_type: outputType,
        top_k: topK,
        enable_reranking: rerank,
        include_chunks: true,
        llm_provider: provider === "auto" ? undefined : provider,
      });
      onResult(resp);
    } catch (e) {
      alert(String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="query-panel">
      <textarea
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Ask your documents…"
        rows={3}
      />
      <div className="controls">
        <label>
          Output
          <select
            value={outputType}
            onChange={(e) => setOutputType(e.target.value as OutputType)}
          >
            <option value="answer">Answer</option>
            <option value="summary">Summary</option>
            <option value="decisions">Decisions</option>
            <option value="risks">Risks</option>
            <option value="actions">Actions</option>
          </select>
        </label>
        <label>
          Top-k
          <input
            type="number" min={1} max={20}
            value={topK}
            onChange={(e) => setTopK(parseInt(e.target.value || "5"))}
          />
        </label>
        <label>
          Provider
          <select value={provider} onChange={(e) => setProvider(e.target.value)}>
            <option value="auto">auto</option>
            <option value="ollama">ollama</option>
            <option value="openai">openai</option>
            <option value="anthropic">anthropic</option>
          </select>
        </label>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={rerank}
            onChange={(e) => setRerank(e.target.checked)}
          />
          Reranking
        </label>
        <button className="btn primary" onClick={submit} disabled={loading}>
          {loading ? "Running…" : "Run"}
        </button>
      </div>
    </section>
  );
}
