/** The only module that knows a network exists.
 *
 *  Talks to the FastAPI backend in the parent directory (api.py). UI code never
 *  imports fetch or a URL. */
import type { AskRequest, Model, Source, StreamEvent } from "../types";
import { decodeFrame, framePayload, splitFrames } from "./sse";

const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");
export const API_BASE = BASE || location.origin;
export const APP_NAME = import.meta.env.VITE_APP_NAME || "HFrag";

export class ApiError extends Error {}

/** FastAPI puts the reason in `detail`. Showing it beats "the server returned
 *  400" when the reason is something the user can act on. */
async function failure(res: Response): Promise<ApiError> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return new ApiError(body.detail);
  } catch {
    /* not JSON */
  }
  return new ApiError(`The server returned ${res.status}.`);
}

/* ------------------------------------------------------------------ models */

/** The backend's "models" are the RAG versions — structured vs normal chunking.
 *  Picking one picks the database the answer is retrieved from. */
export async function getModels(): Promise<Model[]> {
  let res: Response;
  try {
    res = await fetch(BASE + "/models");
  } catch {
    throw new ApiError("Cannot reach the RAG server.");
  }
  if (!res.ok) throw await failure(res);
  const data = (await res.json()) as { models: Model[] } | Model[];
  return Array.isArray(data) ? data : data.models;
}

/* -------------------------------------------------------------------- chat */

/** Read an SSE body into typed events.
 *
 *  Frames from POST /chat:
 *    data: {"sources": [...]}
 *    data: {"token": "..."}       (or {"delta": "..."})
 *    data: [DONE]
 *  Framing and decoding live in ./sse.ts so they can be checked directly. */
async function* readSse(res: Response, signal?: AbortSignal): AsyncGenerator<StreamEvent> {
  const reader = res.body?.getReader();
  if (!reader) throw new ApiError("The server sent an empty response.");
  const decoder = new TextDecoder();
  let buf = "";

  try {
    while (!signal?.aborted) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      const { frames, rest } = splitFrames(buf);
      buf = rest;
      for (const frame of frames)
        for (const ev of decodeFrame(framePayload(frame))) {
          if (ev.type === "done") return yield ev;
          yield ev;
        }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
  yield { type: "done" };
}

export async function* ask(req: AskRequest): AsyncGenerator<StreamEvent> {
  let res: Response;
  try {
    res = await fetch(BASE + "/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      signal: req.signal,
      body: JSON.stringify({
        conversation_id: req.conversationId,
        message: req.message,
        model: req.model,
        history: req.history,
        top_k: req.settings.topK,
        min_score: req.settings.minScore,
        stream: true,
      }),
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    throw new ApiError("Cannot reach the RAG server.");
  }
  if (!res.ok) throw await failure(res);

  // A backend that ignores stream:true returns one JSON body. Handle both.
  if (!res.headers.get("content-type")?.includes("event-stream")) {
    const body = (await res.json()) as { answer: string; sources?: Source[] };
    if (body.sources) yield { type: "sources", sources: body.sources };
    yield { type: "token", text: body.answer };
    return yield { type: "done" };
  }
  yield* readSse(res, req.signal);
}

/* ------------------------------------------------------------------- title */

/** Conversation titles are derived locally — a title is not worth a round trip
 *  or a second model call. */
export function titleFrom(question: string): string {
  const clean = question.replace(/\s+/g, " ").trim().replace(/[?.!,;:]+$/, "");
  const words = clean.split(" ");
  const short = words.slice(0, 7).join(" ") + (words.length > 7 ? "…" : "");
  return short.charAt(0).toUpperCase() + short.slice(1);
}
