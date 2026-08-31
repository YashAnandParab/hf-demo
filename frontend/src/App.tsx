import { useEffect, useLayoutEffect, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { PanelLeft, Plus, RotateCw, SquareLibrary } from "lucide-react";
import { APP_NAME, getModels } from "./lib/api";
import {
  activeConversation,
  applyTheme,
  useChat,
  useModels,
  useUi,
} from "./store";
import { Composer } from "./components/Composer";
import { MessageView } from "./components/Message";
import { Settings } from "./components/Settings";
import { Sidebar } from "./components/Sidebar";
import { SourcePanel } from "./components/SourcePanel";
import { IconButton, Skeleton } from "./components/ui";

const STARTERS = [
  "Why does more choice make investing harder?",
  "What does “return free risk” mean?",
  "Where does a real advisor actually add value?",
  "How does recency distort the case for gold?",
];

/* ------------------------------------------------------------ empty state */

function EmptyState() {
  const setQuery = useChat((s) => s.setQuery);
  const send = useChat((s) => s.send);

  return (
    <div className="flex min-h-full flex-col items-center justify-center px-4 py-10">
      <div className="w-full max-w-[46rem]">
        <div className="rise mb-7 text-center">
          <svg
            viewBox="0 0 24 24"
            width="30"
            height="30"
            aria-hidden
            className="mx-auto"
          >
            <path
              d="M12 2.6 14 9.9 21.4 12 14 14.1 12 21.4 10 14.1 2.6 12 10 9.9Z"
              fill="var(--color-brass)"
            />
          </svg>
          <h1 className="mt-4 text-[26px] leading-tight font-semibold tracking-[-0.02em] text-ink">
            Ask your knowledge base
          </h1>
          <p className="mx-auto mt-2 max-w-[30rem] text-[14.5px] leading-relaxed text-ink-2">
            {APP_NAME} answers from your indexed documents, and shows you which
            passages it used.
          </p>
        </div>

        <Composer centered />

        <ul className="mx-auto mt-6 grid w-full max-w-[46rem] gap-2 sm:grid-cols-2">
          {STARTERS.map((s, i) => (
            <li key={s} className="rise" style={{ animationDelay: `${60 + i * 45}ms` }}>
              <button
                onClick={() => {
                  setQuery("");
                  void send(s);
                }}
                className="w-full rounded-xl border border-line bg-surface px-3.5 py-3 text-left text-[13px] leading-snug text-ink-2 transition-colors duration-150 hover:border-line-strong hover:bg-raised hover:text-ink"
              >
                {s}
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

/* --------------------------------------------------------------- messages */

function Thread() {
  const conv = useChat(activeConversation);
  const generating = useChat((s) => s.generating);
  const scroller = useRef<HTMLDivElement>(null);
  const pinned = useRef(true);
  const count = conv?.messages.length ?? 0;
  const lastLen = conv?.messages.at(-1)?.content.length ?? 0;

  // Follow the stream, but stop fighting the user the moment they scroll up.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && pinned.current)
      el.scrollTo({ top: el.scrollHeight, behavior: count > 1 ? "smooth" : "auto" });
  }, [count, lastLen]);

  const onScroll = () => {
    const el = scroller.current;
    if (el) pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 96;
  };

  if (!conv || conv.messages.length === 0)
    return (
      <div ref={scroller} className="scroll-quiet flex-1 overflow-y-auto">
        <EmptyState />
      </div>
    );

  return (
    <>
      <div
        ref={scroller}
        onScroll={onScroll}
        className="scroll-quiet flex-1 overflow-y-auto"
      >
        <div className="mx-auto w-full max-w-[46rem] px-4 pt-4 pb-6 sm:px-6">
          {conv.messages.map((m) => (
            <MessageView key={m.id} msg={m} />
          ))}
        </div>
      </div>
      <Composer />
      <span className="sr-only" aria-live="polite">
        {generating ? "Generating an answer" : ""}
      </span>
    </>
  );
}

/* ------------------------------------------------------------------- app */

export default function App() {
  const { sidebarOpen, toggleSidebar, theme, source } = useUi();
  const conv = useChat(activeConversation);
  const newChat = useChat((s) => s.newChat);
  const setModels = useModels((s) => s.setModels);

  const models = useQuery({
    queryKey: ["models"],
    queryFn: getModels,
    staleTime: 5 * 60_000,
    retry: 1,
  });

  useEffect(() => {
    if (models.data) setModels(models.data);
  }, [models.data, setModels]);

  useEffect(() => {
    applyTheme(theme);
    if (theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => applyTheme("system");
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, [theme]);

  // Start narrow screens with the drawer shut.
  useEffect(() => {
    if (window.matchMedia("(max-width: 767px)").matches) toggleSidebar(false);
  }, [toggleSidebar]);

  return (
    <div className="flex h-[100dvh] overflow-hidden bg-canvas">
      <Sidebar />

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-13 shrink-0 items-center gap-2 border-b border-line px-3 sm:px-4">
          {!sidebarOpen && (
            <IconButton
              label="Open sidebar"
              side="bottom"
              onClick={() => toggleSidebar(true)}
            >
              <PanelLeft size={17} />
            </IconButton>
          )}
          <h1 className="min-w-0 flex-1 truncate text-[13.5px] font-medium text-ink">
            {conv?.messages.length ? conv.title : APP_NAME}
          </h1>
          {models.isError && (
            <button
              onClick={() => void models.refetch()}
              className="flex items-center gap-1.5 rounded-lg border border-danger/40 px-2 py-1 text-[11.5px] text-danger"
            >
              <RotateCw size={12} />
              Models unavailable — retry
            </button>
          )}
          {models.isLoading && <Skeleton className="h-5 w-20" />}
          <IconButton label="New chat" side="bottom" onClick={newChat}>
            <Plus size={17} />
          </IconButton>
          <IconButton
            label={source ? "Hide sources" : "Sources"}
            side="bottom"
            active={!!source}
            className="lg:inline-flex"
            onClick={() => useUi.getState().openSource(null)}
            disabled={!source}
          >
            <SquareLibrary size={17} />
          </IconButton>
        </header>

        <Thread />
      </main>

      <SourcePanel />
      <Settings />
    </div>
  );
}
