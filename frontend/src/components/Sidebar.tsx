import { useEffect, useMemo, useRef, useState } from "react";
import {
  Archive,
  ArchiveRestore,
  MoreHorizontal,
  PanelLeftClose,
  Pencil,
  Plus,
  Search,
  Settings,
  Trash2,
  X,
} from "lucide-react";
import { APP_NAME } from "../lib/api";
import {
  groupByDate,
  useChat,
  useUi,
} from "../store";
import type { Conversation } from "../types";
import { IconButton, Popover, cx } from "./ui";

/** Wrap the matched span so a search result explains itself. */
function Highlight({ text, q }: { text: string; q: string }) {
  const i = q ? text.toLowerCase().indexOf(q.toLowerCase()) : -1;
  if (i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark className="bg-brass/25 text-ink rounded-[3px] px-px">
        {text.slice(i, i + q.length)}
      </mark>
      {text.slice(i + q.length)}
    </>
  );
}

function Row({ conv, query }: { conv: Conversation; query: string }) {
  const activeId = useChat((s) => s.activeId);
  const open = useChat((s) => s.open);
  const rename = useChat((s) => s.rename);
  const remove = useChat((s) => s.remove);
  const archive = useChat((s) => s.archive);
  const [menu, setMenu] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(conv.title);
  const inputRef = useRef<HTMLInputElement>(null);
  const active = activeId === conv.id;

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const commit = () => {
    rename(conv.id, draft);
    setEditing(false);
  };

  if (editing)
    return (
      <input
        ref={inputRef}
        value={draft}
        aria-label="Conversation title"
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") commit();
          if (e.key === "Escape") {
            setDraft(conv.title);
            setEditing(false);
          }
        }}
        className="w-full rounded-lg border border-brass/60 bg-raised px-2.5 py-2 text-[13.5px] text-ink outline-none"
      />
    );

  return (
    <div className="group/row relative">
      <button
        onClick={() => open(conv.id)}
        className={cx(
          "flex w-full items-center gap-2 rounded-lg py-2 pr-8 pl-2.5 text-left transition-colors duration-150",
          active ? "bg-raised text-ink" : "text-ink-2 hover:bg-raised/60 hover:text-ink",
        )}
      >
        {/* Active marker: a rule, not a filled block. */}
        <span
          aria-hidden
          className={cx(
            "h-3.5 w-px shrink-0 rounded-full transition-colors",
            active ? "bg-brass" : "bg-transparent",
          )}
        />
        <span className="truncate text-[13.5px] leading-5">
          <Highlight text={conv.title} q={query} />
        </span>
      </button>

      <div
        className={cx(
          "absolute top-1/2 right-1 -translate-y-1/2 transition-opacity duration-150",
          menu || active
            ? "opacity-100"
            : "opacity-0 group-hover/row:opacity-100 group-focus-within/row:opacity-100",
        )}
      >
        <IconButton
          label="More"
          side="bottom"
          className="h-7 w-7"
          aria-expanded={menu}
          onClick={() => setMenu((v) => !v)}
        >
          <MoreHorizontal size={15} />
        </IconButton>
        <Popover open={menu} onClose={() => setMenu(false)} className="top-8 right-0 w-40 p-1">
          {[
            {
              icon: <Pencil size={14} />,
              label: "Rename",
              run: () => {
                setDraft(conv.title);
                setEditing(true);
              },
            },
            {
              icon: conv.archived ? <ArchiveRestore size={14} /> : <Archive size={14} />,
              label: conv.archived ? "Unarchive" : "Archive",
              run: () => archive(conv.id),
            },
            {
              icon: <Trash2 size={14} />,
              label: "Delete",
              run: () => remove(conv.id),
              danger: true,
            },
          ].map((item) => (
            <button
              key={item.label}
              onClick={() => {
                item.run();
                setMenu(false);
              }}
              className={cx(
                "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] transition-colors",
                item.danger
                  ? "text-danger hover:bg-danger/10"
                  : "text-ink-2 hover:bg-surface hover:text-ink",
              )}
            >
              {item.icon}
              {item.label}
            </button>
          ))}
        </Popover>
      </div>
    </div>
  );
}

export function Sidebar() {
  const conversations = useChat((s) => s.conversations);
  const newChat = useChat((s) => s.newChat);
  const { sidebarOpen, toggleSidebar, setSettingsOpen } = useUi();
  const [query, setQuery] = useState("");
  const [showArchived, setShowArchived] = useState(false);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    const list = conversations.filter(
      (c) =>
        Boolean(c.archived) === showArchived &&
        c.messages.length > 0 &&
        (!q || c.title.toLowerCase().includes(q)),
    );
    return groupByDate(list);
  }, [conversations, query, showArchived]);

  const empty = groups.length === 0;

  return (
    <>
      {/* Scrim for the mobile drawer. */}
      <div
        onClick={() => toggleSidebar(false)}
        className={cx(
          "fixed inset-0 z-30 bg-black/50 transition-opacity duration-200 md:hidden",
          sidebarOpen ? "opacity-100" : "pointer-events-none opacity-0",
        )}
      />
      <aside
        className={cx(
          "fixed inset-y-0 left-0 z-40 flex w-[274px] shrink-0 flex-col border-r border-line bg-surface",
          "transition-transform duration-200 ease-[cubic-bezier(0.22,1,0.36,1)]",
          "md:static md:translate-x-0",
          sidebarOpen ? "translate-x-0" : "-translate-x-full md:hidden",
        )}
      >
        <header className="flex items-center gap-2 px-3 pt-3 pb-1">
          <div className="flex flex-1 items-center gap-2 px-1.5">
            <svg viewBox="0 0 24 24" width="17" height="17" aria-hidden>
              <path
                d="M12 2.6 14 9.9 21.4 12 14 14.1 12 21.4 10 14.1 2.6 12 10 9.9Z"
                fill="var(--color-brass)"
              />
            </svg>
            <span className="text-[14px] font-semibold tracking-[-0.01em] text-ink">
              {APP_NAME}
            </span>
          </div>
          <IconButton
            label="Close sidebar"
            side="bottom"
            onClick={() => toggleSidebar(false)}
          >
            <PanelLeftClose size={17} />
          </IconButton>
        </header>

        <div className="space-y-1.5 px-3 pt-2 pb-3">
          <button
            onClick={newChat}
            className="flex w-full items-center gap-2 rounded-xl border border-line-strong bg-raised px-3 py-2.5 text-[13.5px] font-medium text-ink transition-colors duration-150 hover:border-brass/45"
          >
            <Plus size={16} className="text-brass" />
            New chat
          </button>

          <div className="relative">
            <Search
              size={14}
              className="pointer-events-none absolute top-1/2 left-3 -translate-y-1/2 text-ink-3"
            />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search conversations"
              aria-label="Search conversations"
              className="w-full rounded-xl bg-raised/60 py-2 pr-8 pl-[34px] text-[13px] text-ink placeholder:text-ink-3 outline-none transition-colors focus:bg-raised"
            />
            {query && (
              <button
                onClick={() => setQuery("")}
                aria-label="Clear search"
                className="absolute top-1/2 right-2 -translate-y-1/2 rounded p-1 text-ink-3 hover:text-ink"
              >
                <X size={13} />
              </button>
            )}
          </div>
        </div>

        <nav className="scroll-quiet flex-1 overflow-y-auto px-3 pb-2">
          {empty ? (
            <p className="px-2.5 py-6 text-[13px] leading-relaxed text-ink-3">
              {query
                ? `Nothing matches “${query}”.`
                : showArchived
                  ? "No archived conversations."
                  : "Your conversations will appear here. Ask something to begin."}
            </p>
          ) : (
            groups.map(([label, items]) => (
              <section key={label} className="mb-4">
                <h2 className="px-2.5 pt-2 pb-1.5 text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase">
                  {label}
                </h2>
                <div className="space-y-0.5">
                  {items.map((c) => (
                    <Row key={c.id} conv={c} query={query} />
                  ))}
                </div>
              </section>
            ))
          )}
        </nav>

        <footer className="border-t border-line px-3 py-2.5">
          <button
            onClick={() => setShowArchived((v) => !v)}
            className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] text-ink-2 transition-colors hover:bg-raised hover:text-ink"
          >
            <Archive size={15} />
            {showArchived ? "Back to conversations" : "Archived"}
          </button>
          <button
            onClick={() => setSettingsOpen(true)}
            className="flex w-full items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-[13px] text-ink-2 transition-colors hover:bg-raised hover:text-ink"
          >
            <Settings size={15} />
            Settings
          </button>
          <div className="mt-1.5 flex items-center gap-2.5 rounded-xl px-2.5 py-2">
            <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-brass text-[11px] font-semibold text-brass-ink">
              H
            </span>
            <span className="min-w-0">
              <span className="block truncate text-[13px] text-ink">Harsh</span>
              <span className="block text-[11px] text-ink-3">
                {conversations.filter((c) => c.messages.length).length} conversations
              </span>
            </span>
          </div>
        </footer>
      </aside>
    </>
  );
}
