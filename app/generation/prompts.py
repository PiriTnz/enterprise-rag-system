"""Prompt templates.

Every template here is designed for a single principle: the model must
ground its output in the provided context and cite sources by chunk id.
Anything not in the context is "unknown" — not fabricated.
"""
from __future__ import annotations

from textwrap import dedent

from app.schemas.models import RetrievedChunk, StructuredOutputType


SYSTEM_GROUNDED = dedent("""
You are an enterprise knowledge assistant. You answer ONLY from the provided context.

Strict rules:
- If the answer is not in the context, say exactly: "The available documents do not contain enough information to answer this."
- Cite sources inline using [chunk_id] markers after each fact.
- Be concise and factual. No speculation.
- If multiple chunks support a fact, cite the most relevant one.
- Use the same language as the user's query.
""").strip()


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks for the prompt."""
    lines: list[str] = []
    for c in chunks:
        meta = c.chunk.metadata
        header_parts = [f"chunk_id={c.chunk.chunk_id}", f"source={meta.source}"]
        if meta.page is not None:
            header_parts.append(f"page={meta.page}")
        if meta.section:
            header_parts.append(f"section={meta.section}")
        header = " | ".join(header_parts)
        lines.append(f"[{header}]\n{c.chunk.text}")
    return "\n\n---\n\n".join(lines)


def build_answer_prompt(query: str, chunks: list[RetrievedChunk]) -> str:
    return dedent(f"""
    # Context
    {format_context(chunks)}

    # Question
    {query}

    # Instructions
    Answer the question using ONLY the context above.
    After each factual statement, cite the supporting chunk_id in square brackets, e.g. [{chunks[0].chunk.chunk_id if chunks else 'chunk_id'}].
    If the context is insufficient, say so explicitly.
    """).strip()


# ---- Structured outputs ----

_STRUCTURED_INSTRUCTIONS = {
    StructuredOutputType.SUMMARY: dedent("""
    Produce an executive summary covering the key points.
    Return JSON with this shape:
    {
      "summary": "<2-3 sentence overall summary>",
      "items": [
        {"title": "<short heading>", "detail": "<1-2 sentence detail>", "chunk_ids": ["<id1>", ...]},
        ...
      ]
    }
    """).strip(),

    StructuredOutputType.DECISIONS: dedent("""
    Extract decisions explicitly made in the context.
    Return JSON:
    {
      "summary": "<one-line overview>",
      "items": [
        {"title": "<decision>", "detail": "<context, rationale>", "chunk_ids": [...]},
        ...
      ]
    }
    If no decisions are present, return items: [].
    """).strip(),

    StructuredOutputType.RISKS: dedent("""
    Extract risks, blockers, or concerns from the context.
    Return JSON:
    {
      "summary": "<one-line risk landscape>",
      "items": [
        {"title": "<risk>", "detail": "<impact, area>", "chunk_ids": [...]},
        ...
      ]
    }
    """).strip(),

    StructuredOutputType.ACTIONS: dedent("""
    Extract recommended actions, action items, or next steps.
    Return JSON:
    {
      "summary": "<one-line overview>",
      "items": [
        {"title": "<action>", "detail": "<owner, deadline if any>", "chunk_ids": [...]},
        ...
      ]
    }
    """).strip(),
}


def build_structured_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    output_type: StructuredOutputType,
) -> str:
    if output_type == StructuredOutputType.ANSWER:
        return build_answer_prompt(query, chunks)
    instruction = _STRUCTURED_INSTRUCTIONS[output_type]
    return dedent(f"""
    # Context
    {format_context(chunks)}

    # User intent
    {query}

    # Task
    {instruction}

    Use ONLY the context. For every item, include chunk_ids that support it.
    Return ONLY valid JSON, no prose before or after.
    """).strip()
