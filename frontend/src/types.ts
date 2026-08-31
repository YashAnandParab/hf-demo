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
  /** Rerank / similarity score, 0..1. */
  score?: number;
  /** Which retrieval arms surfaced it: vector, fulltext, questions. */
  arms?: string[];
}

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
