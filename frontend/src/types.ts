/** Wire types, mirroring what api.py sends. */

/** A retrieval version — "Structured RAG" or "Normal chunking". The UI calls
 *  these models because that is the slot they sit in; picking one picks the
 *  database the answer comes from. */
export interface Model {
  id: string;
  name: string;
  provider?: string;
  description?: string;
  capabilities?: string[];
}

/** Authoritative evidence vs. the illustrative story attached to it. */
export type SourceKind = "knowledge" | "story";

export interface Source {
  /** The label the answer cites: K1/S1 structured, P1 flat. */
  id: string;
  /** Human title of the containing article/document. */
  document: string;
  /** Original URL, when the corpus was crawled. */
  url?: string;
  page?: number;
  chunk?: number;
  content: string;
  /** Rerank / similarity score, 0..1. Stories are attached, not scored. */
  score?: number;
  /** Which retrieval arms surfaced it: vector, fulltext, questions. */
  arms?: string[];
  /** Set by the structured backend; older payloads fall back to the id prefix. */
  kind?: SourceKind;
  /** Story only: the [K*] labels it illustrates, e.g. "K1, K3". */
  illustrates?: string;
}

/** A story chunk reached the answer because a retrieved knowledge chunk cites
 *  it, so it is evidence of a different kind and is shown as such. */
export const sourceKind = (s: Source): SourceKind =>
  s.kind ?? (s.id.startsWith("S") ? "story" : "knowledge");

export const isStory = (s: Source) => sourceKind(s) === "story";

export type Role = "user" | "assistant";

export interface Message {
  id: string;
  role: Role;
  content: string;
  sources?: Source[];
  model?: string;
  createdAt: string;
  isStreaming?: boolean;
  error?: string;
  vote?: "up" | "down";
}

export interface Conversation {
  id: string;
  title: string;
  model?: string;
  createdAt: string;
  updatedAt: string;
  archived?: boolean;
  messages: Message[];
}

export interface RagSettings {
  topK: number;
  minScore: number;
}

/** One event from the answer stream. */
export type StreamEvent =
  | { type: "sources"; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done" };

export interface AskRequest {
  conversationId: string;
  message: string;
  model: string;
  history: { role: Role; content: string }[];
  settings: RagSettings;
  signal?: AbortSignal;
}
