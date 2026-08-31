import { useEffect, useRef, useState } from "react";
import { ArrowUp, Check, ChevronDown, Square } from "lucide-react";
import { selectedModel, useChat, useModels } from "../store";
import { Popover, Skeleton, cx } from "./ui";

/* ------------------------------------------------------- version selector */

/** Picks the RAG version the question is answered from: structured (knowledge
 *  chunks with stories attached) or normal (flat chunks). It sits where a model
 *  picker usually sits because that is what it is — the choice that changes the
 *  answer. */
function ModelSelector() {
  const { models, selectedId, select } = useModels();
  const current = useModels(selectedModel);
  const [open, setOpen] = useState(false);

  if (!models.length) return <Skeleton className="h-7 w-32 rounded-lg" />;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className={cx(
          "flex items-center gap-1.5 rounded-lg py-1.5 pr-1.5 pl-2.5 text-[12.5px] font-medium transition-colors duration-150",
          open ? "bg-raised text-ink" : "text-ink-2 hover:bg-raised hover:text-ink",
        )}
      >
        <span className="max-w-[9.5rem] truncate sm:max-w-none">
          {current?.name}
        </span>
        <ChevronDown
          size={13}
          className={cx("transition-transform duration-150", open && "rotate-180")}
        />
      </button>

      <Popover
        open={open}
        onClose={() => setOpen(false)}
        className="bottom-full left-0 mb-2 w-[19rem] p-1.5"
      >
        <p className="px-2.5 pt-1.5 pb-2 text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase">
          Retrieval version
        </p>
        <ul role="listbox" aria-label="Retrieval version">
          {models.map((m) => {
            const on = m.id === selectedId;
            return (
              <li key={m.id}>
                <button
                  role="option"
                  aria-selected={on}
                  onClick={() => {
                    select(m.id);
                    setOpen(false);
                  }}
                  className={cx(
                    "flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors duration-150",
                    on ? "bg-surface" : "hover:bg-surface",
                  )}
                >
                  <Check
                    size={14}
                    className={cx("mt-0.5 shrink-0 text-brass", !on && "invisible")}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="flex items-baseline gap-2">
                      <span className="truncate text-[13.5px] font-medium text-ink">
                        {m.name}
                      </span>
                      <span className="shrink-0 truncate text-[10.5px] text-ink-3">
                        {m.provider}
                      </span>
                    </span>
                    <span className="mt-0.5 block text-[12px] leading-snug text-ink-2">
                      {m.description}
                    </span>
                    {!!m.capabilities?.length && (
                      <span className="mt-1.5 flex flex-wrap gap-1">
                        {m.capabilities.map((c) => (
                          <span
                            key={c}
                            className="rounded border border-line-strong px-1.5 py-px font-mono text-[10px] tracking-wide text-ink-3"
                          >
                            {c}
                          </span>
                        ))}
                      </span>
                    )}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </Popover>
    </div>
  );
}

/* -------------------------------------------------------------- composer */

export function Composer({ centered }: { centered?: boolean }) {
  // Selectors, not the whole store: this component owns the text input and
  // must not re-render on every streamed token.
  const send = useChat((s) => s.send);
  const stop = useChat((s) => s.stop);
  const generating = useChat((s) => s.generating);
  const query = useChat((s) => s.query);
  const setQuery = useChat((s) => s.setQuery);
  const area = useRef<HTMLTextAreaElement>(null);

  // Grow to fit, then scroll. Reset first so deleting text shrinks it again.
  useEffect(() => {
    const el = area.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 224) + "px";
  }, [query]);

  // New chat / chat switch lands the caret in the composer, on pointer devices
  // only — stealing focus on mobile pops the keyboard over the whole screen.
  const activeId = useChat((s) => s.activeId);
  useEffect(() => {
    if (window.matchMedia("(hover: hover)").matches) area.current?.focus();
  }, [activeId]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        area.current?.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  const submit = () => {
    const text = query.trim();
    if (!text || generating) return;
    setQuery("");
    void send(text);
  };

  return (
    <div className={cx("w-full", centered ? "" : "px-4 pb-4 sm:px-6")}>
      <div className="mx-auto w-full max-w-[46rem]">
        <div
          className={cx(
            "rounded-2xl border border-line bg-surface transition-[border-color,box-shadow] duration-200",
            "shadow-[0_1px_2px_rgba(0,0,0,0.18),0_12px_28px_-14px_rgba(0,0,0,0.5)]",
            "focus-within:border-line-strong focus-within:shadow-[0_1px_2px_rgba(0,0,0,0.2),0_18px_38px_-16px_rgba(0,0,0,0.6)]",
          )}
        >
          <label htmlFor="prompt" className="sr-only">
            Ask a question
          </label>
          <textarea
            id="prompt"
            ref={area}
            rows={1}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                e.preventDefault();
                submit();
              }
            }}
            placeholder="Ask anything about your knowledge base…"
            className="scroll-quiet block max-h-56 w-full resize-none bg-transparent px-4 pt-3.5 pb-1 text-[15px] leading-relaxed text-ink placeholder:text-ink-3 outline-none"
          />

          <div className="flex items-center gap-1 px-2.5 pt-1 pb-2.5">
            <ModelSelector />
            <span className="flex-1" />
            {generating ? (
              <button
                onClick={stop}
                aria-label="Stop generating"
                className="grid h-8 w-8 place-items-center rounded-lg border border-line-strong text-ink transition-colors hover:bg-raised"
              >
                <Square size={12} fill="currentColor" />
              </button>
            ) : (
              <button
                onClick={submit}
                disabled={!query.trim()}
                aria-label="Send"
                className={cx(
                  "grid h-8 w-8 place-items-center rounded-lg transition-all duration-150",
                  query.trim()
                    ? "bg-brass text-brass-ink hover:brightness-110"
                    : "bg-raised text-ink-3",
                )}
              >
                <ArrowUp size={17} />
              </button>
            )}
          </div>
        </div>

        <p className="mt-2 text-center text-[11px] text-ink-3">
          Answers are generated from your indexed documents. Check the sources.
        </p>
      </div>
    </div>
  );
}
