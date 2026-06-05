import type { QueryResponse } from "../types";

export function AnswerView({
  response,
  loading,
}: {
  response: QueryResponse | null;
  loading: boolean;
}) {
  if (loading) return <div className="answer skeleton">Generating grounded answer…</div>;
  if (!response) return null;

  return (
    <section className="answer">
      <div className="metrics">
        <Metric label="Confidence" value={response.confidence.toFixed(2)} />
        <Metric label="Latency" value={`${Math.round(response.latency_ms)}ms`} />
        <Metric label="Provider" value={response.provider_used} />
        <Metric label="Citations" value={String(response.citations.length)} />
      </div>

      <div className="answer-text">
        <h3>Answer</h3>
        <p>{response.answer}</p>
      </div>

      {response.items.length > 0 && (
        <div className="items">
          <h3>Items</h3>
          {response.items.map((it, i) => (
            <div className="item" key={i}>
              <div className="item-title">{it.title}</div>
              <div className="item-detail">{it.detail}</div>
              <div className="item-citations">
                {it.citations.map((c) => (
                  <span className="chip" key={c.chunk_id}>
                    {c.source}{c.page ? ` p.${c.page}` : ""}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {response.citations.length > 0 && (
        <div className="citations">
          <h3>Sources</h3>
          <ol>
            {response.citations.map((c) => (
              <li key={c.chunk_id}>
                <b>{c.source}</b>{c.page ? `, page ${c.page}` : ""}
                <span className="score"> · {c.score.toFixed(2)}</span>
                <blockquote>{c.snippet}</blockquote>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
    </div>
  );
}
