import { create } from "zustand";
import { persist } from "zustand/middleware";
import { ApiError, ask, titleFrom } from "./lib/api";
import type {
  Conversation,
  Message,
  Model,
  RagSettings,
  Source,
} from "./types";

const uid = () =>
  globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2);
const now = () => new Date().toISOString();

/* ================================================================== ui ==== */

type Theme = "dark" | "light" | "system";

interface UiState {
  sidebarOpen: boolean;
  settingsOpen: boolean;
  source: Source | null;
  theme: Theme;
  settings: RagSettings;
  toggleSidebar: (v?: boolean) => void;
  setSettingsOpen: (v: boolean) => void;
  openSource: (s: Source | null) => void;
  setTheme: (t: Theme) => void;
  setSettings: (p: Partial<RagSettings>) => void;
}

export const useUi = create<UiState>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      settingsOpen: false,
      source: null,
      theme: "dark",
      settings: { topK: 3, minScore: 0 },
      toggleSidebar: (v) =>
        set((s) => ({ sidebarOpen: v ?? !s.sidebarOpen })),
      setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
      openSource: (source) => set({ source }),
      setTheme: (theme) => set({ theme }),
      setSettings: (p) => set((s) => ({ settings: { ...s.settings, ...p } })),
    }),
    {
      name: "hfrag.ui",
      partialize: (s) => ({
        theme: s.theme,
        settings: s.settings,
        sidebarOpen: s.sidebarOpen,
      }),
    },
  ),
);

/** Resolve `system` against the OS and paint the root element. */
export function applyTheme(theme: Theme) {
  const dark =
    theme === "dark" ||
    (theme === "system" &&
      window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("light", !dark);
  document.documentElement.classList.toggle("dark", dark);
}

/* =============================================================== models === */

interface ModelState {
  models: Model[];
  selectedId: string | null;
  setModels: (m: Model[]) => void;
  select: (id: string) => void;
}

export const useModels = create<ModelState>()(
  persist(
    (set, get) => ({
      models: [],
      selectedId: null,
      setModels: (models) =>
        set({
          models,
          selectedId:
            models.some((m) => m.id === get().selectedId) && get().selectedId
              ? get().selectedId
              : (models[0]?.id ?? null),
        }),
      select: (selectedId) => set({ selectedId }),
    }),
    { name: "hfrag.model", partialize: (s) => ({ selectedId: s.selectedId }) },
  ),
);

export const selectedModel = (s: ModelState) =>
  s.models.find((m) => m.id === s.selectedId) ?? s.models[0];

/* ================================================================= chat === */

interface ChatState {
  conversations: Conversation[];
  activeId: string | null;
  query: string;
  generating: boolean;

  setQuery: (q: string) => void;
  newChat: () => void;
  open: (id: string) => void;
  rename: (id: string, title: string) => void;
  remove: (id: string) => void;
  archive: (id: string) => void;
  vote: (msgId: string, v: "up" | "down") => void;
  stop: () => void;
  send: (text: string) => Promise<void>;
  regenerate: () => Promise<void>;
}

let controller: AbortController | null = null;

export const useChat = create<ChatState>()(
  persist(
    (set, get) => {
      /** Mutate one conversation in place, bumping updatedAt. */
      const patch = (id: string, fn: (c: Conversation) => Conversation) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === id ? { ...fn(c), updatedAt: now() } : c,
          ),
        }));

      const patchMessage = (
        convId: string,
        msgId: string,
        fn: (m: Message) => Message,
      ) =>
        patch(convId, (c) => ({
          ...c,
          messages: c.messages.map((m) => (m.id === msgId ? fn(m) : m)),
        }));

      /** Drive one answer into `replyId`. Shared by send and regenerate. */
      const run = async (convId: string, replyId: string) => {
        const conv = get().conversations.find((c) => c.id === convId);
        if (!conv) return;
        const model = useModels.getState().selectedId ?? "";
        controller = new AbortController();
        set({ generating: true });

        try {
          const history = conv.messages
            .filter((m) => m.id !== replyId && !m.error)
            .map((m) => ({ role: m.role, content: m.content }));
          const question = [...history].reverse().find((m) => m.role === "user");

          for await (const ev of ask({
            conversationId: convId,
            message: question?.content ?? "",
            model,
            history: history.slice(0, -1),
            settings: useUi.getState().settings,
            signal: controller.signal,
          })) {
            if (controller.signal.aborted) break;
            if (ev.type === "sources")
              patchMessage(convId, replyId, (m) => ({ ...m, sources: ev.sources }));
            else if (ev.type === "token")
              patchMessage(convId, replyId, (m) => ({
                ...m,
                content: m.content + ev.text,
              }));
          }
          patchMessage(convId, replyId, (m) => ({ ...m, isStreaming: false }));
        } catch (e) {
          const msg =
            e instanceof ApiError
              ? e.message
              : "Something went wrong generating that answer.";
          patchMessage(convId, replyId, (m) => ({
            ...m,
            isStreaming: false,
            error: msg,
          }));
        } finally {
          controller = null;
          set({ generating: false });
        }
      };

      return {
        conversations: [],
        activeId: null,
        query: "",
        generating: false,

        setQuery: (query) => set({ query }),

        newChat: () => {
          get().stop();
          // Reuse an untouched empty chat rather than stacking blanks.
          const blank = get().conversations.find((c) => c.messages.length === 0);
          if (blank) return set({ activeId: blank.id });
          const conv: Conversation = {
            id: uid(),
            title: "New chat",
            createdAt: now(),
            updatedAt: now(),
            messages: [],
          };
          set((s) => ({
            conversations: [conv, ...s.conversations],
            activeId: conv.id,
          }));
        },

        open: (activeId) => {
          get().stop();
          set({ activeId });
        },

        rename: (id, title) =>
          patch(id, (c) => ({ ...c, title: title.trim() || c.title })),

        remove: (id) =>
          set((s) => {
            const conversations = s.conversations.filter((c) => c.id !== id);
            return {
              conversations,
              activeId:
                s.activeId === id ? (conversations[0]?.id ?? null) : s.activeId,
            };
          }),

        archive: (id) => patch(id, (c) => ({ ...c, archived: !c.archived })),

        vote: (msgId, v) => {
          const id = get().activeId;
          if (id)
            patchMessage(id, msgId, (m) => ({
              ...m,
              vote: m.vote === v ? undefined : v,
            }));
        },

        stop: () => {
          controller?.abort();
          controller = null;
          const id = get().activeId;
          if (id)
            patch(id, (c) => ({
              ...c,
              messages: c.messages.map((m) =>
                m.isStreaming ? { ...m, isStreaming: false } : m,
              ),
            }));
          set({ generating: false });
        },

        send: async (text) => {
          if (!text.trim() || get().generating) return;
          if (!get().activeId) get().newChat();
          const convId = get().activeId!;

          const user: Message = {
            id: uid(),
            role: "user",
            content: text.trim(),
            createdAt: now(),
          };
          const reply: Message = {
            id: uid(),
            role: "assistant",
            content: "",
            model: useModels.getState().selectedId ?? undefined,
            createdAt: now(),
            isStreaming: true,
          };

          patch(convId, (c) => ({
            ...c,
            // First question names the conversation.
            title: c.messages.length ? c.title : titleFrom(user.content),
            messages: [...c.messages, user, reply],
          }));
          await run(convId, reply.id);
        },

        regenerate: async () => {
          const convId = get().activeId;
          if (!convId || get().generating) return;
          const conv = get().conversations.find((c) => c.id === convId);
          const last = conv?.messages.at(-1);
          if (!last || last.role !== "assistant") return;

          patchMessage(convId, last.id, (m) => ({
            ...m,
            content: "",
            sources: undefined,
            error: undefined,
            vote: undefined,
            isStreaming: true,
            model: useModels.getState().selectedId ?? undefined,
          }));
          await run(convId, last.id);
        },
      };
    },
    {
      name: "hfrag.chat",
      partialize: (s) => ({
        // Never persist a half-streamed message as still streaming.
        conversations: s.conversations.map((c) => ({
          ...c,
          messages: c.messages.map((m) => ({ ...m, isStreaming: false })),
        })),
        activeId: s.activeId,
      }),
    },
  ),
);

export const activeConversation = (s: ChatState) =>
  s.conversations.find((c) => c.id === s.activeId) ?? null;

/* --------------------------------------------------------------- grouping */

const DAY = 86_400_000;

export function groupByDate(list: Conversation[]) {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const t0 = start.getTime();
  const buckets: Record<string, Conversation[]> = {};
  const order = ["Today", "Yesterday", "Previous 7 days", "Older"];

  for (const c of [...list].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt))) {
    const at = new Date(c.updatedAt).getTime();
    const key =
      at >= t0
        ? "Today"
        : at >= t0 - DAY
          ? "Yesterday"
          : at >= t0 - 7 * DAY
            ? "Previous 7 days"
            : "Older";
    (buckets[key] ??= []).push(c);
  }
  return order.filter((k) => buckets[k]).map((k) => [k, buckets[k]] as const);
}
