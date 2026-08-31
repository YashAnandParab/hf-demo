/** SSE framing, kept pure so it can be checked without a browser.
 *
 *  Network chunks split anywhere — mid-frame, mid-word, mid-UTF8. Everything
 *  here is about not losing a token at a chunk boundary. */
import type { Source, StreamEvent } from "../types";

/** Split a buffer into complete SSE frames plus the leftover partial frame.
 *  A frame ends at a blank line; anything after the last blank line is
 *  incomplete and must be carried into the next read. */
export function splitFrames(buffer: string): { frames: string[]; rest: string } {
  const parts = buffer.split(/\r?\n\r?\n/);
  return { frames: parts.slice(0, -1), rest: parts.at(-1) ?? "" };
}

/** The `data:` payload of one frame, with multi-line data rejoined. */
export function framePayload(frame: string): string {
  return frame
    .split(/\r?\n/)
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(5).trim())
    .join("\n");
}

/** Decode one payload into events. Unrecognised shapes yield nothing rather
 *  than throwing — a stray keep-alive must not kill an answer mid-stream. */
export function decodeFrame(payload: string): StreamEvent[] {
  if (!payload) return [];
  if (payload === "[DONE]") return [{ type: "done" }];

  let obj: Record<string, unknown>;
  try {
    obj = JSON.parse(payload) as Record<string, unknown>;
  } catch {
    // Some backends stream bare text frames.
    return [{ type: "token", text: payload }];
  }

  const out: StreamEvent[] = [];
  if (Array.isArray(obj.sources))
    out.push({ type: "sources", sources: obj.sources as Source[] });
  const text = obj.token ?? obj.delta ?? obj.content;
  if (typeof text === "string" && text) out.push({ type: "token", text });
  if (obj.done) out.push({ type: "done" });
  return out;
}
