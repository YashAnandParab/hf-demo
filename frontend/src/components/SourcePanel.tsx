import { useEffect } from "react";
import { BookOpen, ExternalLink, X } from "lucide-react";
import { useUi } from "../store";
import { isStory } from "../types";
import { cx } from "./ui";

/** Right-hand evidence panel. Desktop: a resident column. Mobile: a drawer
 *  over the conversation. */
export function SourcePanel() {
  const source = useUi((s) => s.source);
  const close = useUi((s) => s.openSource);

  useEffect(() => {
    if (!source) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close(null);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [source, close]);

  const story = !!source && isStory(source);

  const meta: [string, string][] = source
    ? ([
        ["Document", source.document],
        ["Kind", story ? "Illustrative story" : "Authoritative knowledge"],
        story && source.illustrates && ["Illustrates", source.illustrates],
        source.page != null && ["Page", String(source.page)],
        source.chunk != null && ["Chunk", String(source.chunk)],
        source.score != null && ["Similarity", source.score.toFixed(3)],
        source.arms?.length && ["Retrieved by", source.arms.join(", ")],
      ].filter(Boolean) as [string, string][])
    : [];

  return (
    <>
      <div
        onClick={() => close(null)}
        className={cx(
          "fixed inset-0 z-40 bg-black/50 transition-opacity duration-200 lg:hidden",
          source ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      />
      <aside
        aria-label="Source detail"
        aria-hidden={!source}
        className={cx(
          "fixed inset-y-0 right-0 z-50 flex w-full max-w-[24rem] flex-col border-l border-line bg-surface",
          "transition-transform duration-250 ease-[cubic-bezier(0.22,1,0.36,1)]",
          "lg:static lg:z-0 lg:max-w-none",
          source
            ? "translate-x-0 lg:w-[23rem]"
            : "translate-x-full lg:w-0 lg:overflow-hidden lg:border-l-0",
        )}
      >
        {source && (
          <>
            <header className="flex items-start justify-between gap-3 border-b border-line px-5 py-4">
              <div className="min-w-0">
                <p className="flex items-center gap-1.5 text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase">
                  {story && <BookOpen size={11} />}
                  {story ? "Linked story" : "Source"}
                </p>
                <h2 className="mt-1.5 text-[15px] leading-snug font-medium text-ink">
                  {source.document}
                </h2>
              </div>
              <button
                onClick={() => close(null)}
                aria-label="Close source"
                className="-mt-1 shrink-0 rounded-lg p-1.5 text-ink-3 transition-colors hover:bg-raised hover:text-ink"
              >
                <X size={17} />
              </button>
            </header>

            <div className="scroll-quiet flex-1 overflow-y-auto px-5 py-5">
              <p className="text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase">
                {story ? "Story passage" : "Retrieved passage"}
              </p>
              {/* The quoted chunk is the evidence — give it the brass rule. A
                  story illustrates that evidence, so it gets a plain one. */}
              <blockquote
                className={cx(
                  "mt-2.5 border-l pl-3.5 text-[13.5px] leading-relaxed",
                  story
                    ? "border-line-strong text-ink-2 italic"
                    : "border-brass/60 text-ink",
                )}
              >
                {source.content}
              </blockquote>
              {story && (
                <p className="mt-2.5 text-[12px] leading-relaxed text-ink-3">
                  Attached because{" "}
                  {source.illustrates
                    ? `${source.illustrates} cites it`
                    : "a retrieved knowledge chunk cites it"}
                  . It is context for the answer, not evidence for it.
                </p>
              )}

              <p className="mt-7 text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase">
                Metadata
              </p>
              <dl className="mt-2.5 divide-y divide-line border-y border-line">
                {meta.map(([k, v]) => (
                  <div key={k} className="flex gap-4 py-2">
                    <dt className="w-[6.5rem] shrink-0 text-[12.5px] text-ink-3">
                      {k}
                    </dt>
                    <dd className="min-w-0 flex-1 font-mono text-[12px] leading-relaxed break-words text-ink-2">
                      {v}
                    </dd>
                  </div>
                ))}
              </dl>

              {source.score != null && (
                <div className="mt-4">
                  <div className="flex items-baseline justify-between">
                    <span className="text-[12px] text-ink-3">Similarity</span>
                    <span className="font-mono text-[12px] text-ink">
                      {source.score.toFixed(3)}
                    </span>
                  </div>
                  <div
                    role="meter"
                    aria-valuenow={source.score}
                    aria-valuemin={0}
                    aria-valuemax={1}
                    aria-label="Similarity score"
                    className="mt-1.5 h-1 overflow-hidden rounded-full bg-raised"
                  >
                    <div
                      className="h-full rounded-full bg-brass"
                      style={{ width: `${source.score * 100}%` }}
                    />
                  </div>
                </div>
              )}

              {source.url && (
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="mt-6 inline-flex items-center gap-1.5 text-[12.5px] text-ink-2 underline decoration-line-strong underline-offset-4 transition-colors hover:text-ink"
                >
                  Open original
                  <ExternalLink size={12} />
                </a>
              )}
            </div>
          </>
        )}
      </aside>
    </>
  );
}
