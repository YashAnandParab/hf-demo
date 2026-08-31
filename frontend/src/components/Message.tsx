import { Children, isValidElement, memo, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {
  AlertTriangle,
  BookOpen,
  Check,
  ChevronRight,
  Copy,
  FileText,
  RefreshCw,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { useChat, useModels, useUi } from "../store";
import { isStory } from "../types";
import type { Message as Msg, Source } from "../types";
import { IconButton, cx, useToast } from "./ui";

/* ------------------------------------------------------------- citations */

/** Turn a citation in a text run into a real reference marker.
 *
 *  The backend labels its passages [K1]/[S1] (structured) or [P1] (flat), and
 *  each source carries that label as its id, so the match is by label with a
 *  positional fallback for a bare [1]. Only citations that actually resolve
 *  become interactive — a hallucinated [7] stays plain text. */
function linkCitations(
  node: ReactNode,
  sources: Source[],
  open: (s: Source) => void,
): ReactNode {
  if (typeof node === "string") {
    const parts = node.split(/(\[[KSP]?\d{1,2}\])/g);
    if (parts.length === 1) return node;
    return parts.map((part, i) => {
      const m = /^\[([KSP]?)(\d{1,2})\]$/.exec(part);
      const src = m
        ? (sources.find((s) => s.id === m[1] + m[2]) ??
          (m[1] ? undefined : sources[Number(m[2]) - 1]))
        : undefined;
      if (!src) return part;
      return (
        <button
          key={i}
          onClick={() => open(src)}
          title={src.document}
          className="mx-px inline-flex h-[1.15em] min-w-[1.15em] translate-y-[-0.28em] items-center justify-center rounded-[4px] border border-brass/40 bg-brass/12 px-[0.28em] align-baseline font-mono text-[0.68em] leading-none font-medium text-brass transition-colors hover:bg-brass hover:text-brass-ink"
        >
          {src.id}
        </button>
      );
    });
  }
  if (Array.isArray(node))
    return node.map((c, i) => (
      <span key={i}>{linkCitations(c, sources, open)}</span>
    ));
  return node;
}

/* ------------------------------------------------------------ code block */

function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const child = Children.toArray(children)[0];
  const cls =
    (isValidElement<{ className?: string }>(child) &&
      child.props.className) ||
    "";
  const lang = /language-(\w+)/.exec(cls)?.[1] ?? "text";

  const copy = (e: React.MouseEvent<HTMLButtonElement>) => {
    const pre = e.currentTarget.closest("figure")?.querySelector("pre");
    void navigator.clipboard.writeText(pre?.textContent ?? "");
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <figure className="my-4 overflow-hidden rounded-xl border border-line bg-raised">
      <figcaption className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <span className="font-mono text-[11px] tracking-wide text-ink-3">
          {lang}
        </span>
        <button
          onClick={copy}
          className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-[11px] text-ink-3 transition-colors hover:text-ink"
        >
          {copied ? <Check size={12} className="text-brass" /> : <Copy size={12} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </figcaption>
      <pre className="scroll-quiet">{children}</pre>
    </figure>
  );
}

/* --------------------------------------------------------------- sources */

/** One retrieved passage. Stories carry the quieter treatment: they reached the
 *  answer as illustration, not as evidence, and the card should not claim
 *  otherwise. */
function SourceCard({ source }: { source: Source }) {
  const openSource = useUi((s) => s.openSource);
  const story = isStory(source);
  const Icon = story ? BookOpen : FileText;

  const meta = [
    story && source.illustrates && `illustrates ${source.illustrates}`,
    source.page != null && `p. ${source.page}`,
    source.chunk != null && `chunk ${source.chunk}`,
    source.score != null && `${source.score.toFixed(2)} sim`,
  ]
    .filter(Boolean)
    .join("  ·  ");

  return (
    <li>
      <button
        onClick={() => openSource(source)}
        className={cx(
          "group flex w-full items-start gap-3 rounded-xl border px-3 py-2.5 text-left transition-colors duration-150 hover:border-line-strong hover:bg-raised",
          story ? "border-dashed border-line bg-transparent" : "border-line bg-surface",
        )}
      >
        <span
          className={cx(
            "mt-0.5 grid h-5 min-w-5 shrink-0 place-items-center rounded-md border px-1 font-mono text-[10.5px]",
            story
              ? "border-line-strong bg-raised text-ink-2"
              : "border-brass/35 bg-brass/10 text-brass",
          )}
        >
          {source.id}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex items-center gap-1.5">
            <Icon size={12} className="shrink-0 text-ink-3" />
            <span
              className={cx(
                "truncate text-[13px]",
                story ? "text-ink-2" : "text-ink",
              )}
            >
              {source.document}
            </span>
          </span>
          {meta && (
            <span className="mt-1 block truncate font-mono text-[11px] text-ink-3">
              {meta}
            </span>
          )}
        </span>
        <ChevronRight
          size={14}
          className="mt-1 shrink-0 text-ink-3 transition-transform duration-150 group-hover:translate-x-0.5"
        />
      </button>
    </li>
  );
}

function GroupLabel({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 mb-1.5 text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase first:mt-0">
      {children}
    </p>
  );
}

function SourceList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);

  const knowledge = sources.filter((s) => !isStory(s));
  const stories = sources.filter(isStory);

  return (
    <section className="mt-5 border-t border-line pt-3">
      <button
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="group flex items-center gap-1.5 text-[12px] font-medium text-ink-2 transition-colors hover:text-ink"
      >
        <ChevronRight
          size={13}
          className={cx("transition-transform duration-200", open && "rotate-90")}
        />
        Sources
        <span className="ml-0.5 rounded-full border border-line-strong px-1.5 font-mono text-[10.5px] text-ink-3">
          {knowledge.length}
        </span>
        {/* A story in the answer's context is worth seeing before expanding. */}
        {!!stories.length && (
          <span className="inline-flex items-center gap-1 rounded-full border border-dashed border-line-strong px-1.5 font-mono text-[10.5px] text-ink-3">
            <BookOpen size={10} />
            {stories.length} {stories.length === 1 ? "story" : "stories"}
          </span>
        )}
      </button>

      {open && (
        // Only label the groups when there are two kinds to tell apart — the
        // flat version has nothing to separate.
        <div className="mt-2.5">
          {!!stories.length && <GroupLabel>Knowledge</GroupLabel>}
          <ul className="space-y-1.5">
            {knowledge.map((s) => (
              <SourceCard key={s.id} source={s} />
            ))}
          </ul>
          {!!stories.length && (
            <>
              <GroupLabel>Illustrative stories</GroupLabel>
              <ul className="space-y-1.5">
                {stories.map((s) => (
                  <SourceCard key={s.id} source={s} />
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </section>
  );
}

/* ---------------------------------------------------------------- bubble */

function Actions({ msg }: { msg: Msg }) {
  const regenerate = useChat((s) => s.regenerate);
  const vote = useChat((s) => s.vote);
  const generating = useChat((s) => s.generating);
  const models = useModels((s) => s.models);
  const toast = useToast();
  const modelName = models.find((m) => m.id === msg.model)?.name;

  return (
    <div className="mt-2 flex items-center gap-0.5 opacity-70 transition-opacity duration-150 group-hover/msg:opacity-100 focus-within:opacity-100">
      <IconButton
        label="Copy answer"
        onClick={() => {
          void navigator.clipboard.writeText(msg.content);
          toast("Answer copied");
        }}
      >
        <Copy size={15} />
      </IconButton>
      <IconButton label="Regenerate" disabled={generating} onClick={() => void regenerate()}>
        <RefreshCw size={15} />
      </IconButton>
      <IconButton
        label="Good answer"
        active={msg.vote === "up"}
        onClick={() => vote(msg.id, "up")}
      >
        <ThumbsUp size={15} />
      </IconButton>
      <IconButton
        label="Bad answer"
        active={msg.vote === "down"}
        onClick={() => vote(msg.id, "down")}
      >
        <ThumbsDown size={15} />
      </IconButton>
      {modelName && (
        <span className="ml-1.5 text-[11px] text-ink-3">{modelName}</span>
      )}
    </div>
  );
}

export const MessageView = memo(function MessageView({ msg }: { msg: Msg }) {
  const openSource = useUi((s) => s.openSource);
  const regenerate = useChat((s) => s.regenerate);
  const sources = msg.sources ?? [];

  if (msg.role === "user")
    return (
      <article className="rise flex flex-col items-end gap-1.5 py-3">
        <p className="max-w-[85%] rounded-2xl rounded-br-md bg-raised px-3.5 py-2.5 text-[15px] leading-relaxed whitespace-pre-wrap text-ink">
          {msg.content}
        </p>
      </article>
    );

  const withCites = (children: ReactNode) =>
    linkCitations(children, sources, openSource);

  return (
    <article className="group/msg rise py-3">
      {msg.content ? (
        <div className={cx("prose-answer", msg.isStreaming && "caret")}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[rehypeHighlight]}
            components={{
              pre: CodeBlock,
              p: ({ children }) => <p>{withCites(children)}</p>,
              li: ({ children }) => <li>{withCites(children)}</li>,
              td: ({ children }) => <td>{withCites(children)}</td>,
              a: ({ href, children }) => (
                <a href={href} target="_blank" rel="noreferrer noopener">
                  {children}
                </a>
              ),
            }}
          >
            {msg.content}
          </ReactMarkdown>
        </div>
      ) : msg.isStreaming ? (
        <p className="flex items-center gap-2 text-[13.5px] text-ink-3">
          <span className="inline-flex gap-1" aria-hidden>
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="h-1 w-1 rounded-full bg-brass"
                style={{
                  animation: `caret 1s ${i * 0.15}s var(--ease-out-quint) infinite`,
                }}
              />
            ))}
          </span>
          Searching your documents…
        </p>
      ) : null}

      {msg.error && (
        <div className="mt-2 flex items-start gap-2.5 rounded-xl border border-danger/40 bg-danger/8 px-3 py-2.5">
          <AlertTriangle size={15} className="mt-0.5 shrink-0 text-danger" />
          <div className="min-w-0 flex-1">
            <p className="text-[13px] text-ink">{msg.error}</p>
            <button
              onClick={() => void regenerate()}
              className="mt-1.5 text-[12.5px] font-medium text-brass underline underline-offset-2"
            >
              Try again
            </button>
          </div>
        </div>
      )}

      {!!sources.length && <SourceList sources={sources} />}
      {!msg.isStreaming && !msg.error && msg.content && <Actions msg={msg} />}
    </article>
  );
});
