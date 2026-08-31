import { Monitor, Moon, Sun, X } from "lucide-react";
import { API_BASE } from "../lib/api";
import { useChat, useUi } from "../store";
import { Modal, cx } from "./ui";

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex items-start justify-between gap-6 py-3.5">
      <div className="min-w-0">
        <p className="text-[13.5px] text-ink">{label}</p>
        {hint && <p className="mt-0.5 text-[12px] leading-snug text-ink-3">{hint}</p>}
      </div>
      <div className="shrink-0">{children}</div>
    </div>
  );
}

export function Settings() {
  const { settingsOpen, setSettingsOpen, theme, setTheme, settings, setSettings } =
    useUi();
  const conversations = useChat((s) => s.conversations);

  return (
    <Modal
      open={settingsOpen}
      onClose={() => setSettingsOpen(false)}
      title="Settings"
    >
      <header className="flex items-center justify-between border-b border-line px-5 py-4">
        <h2 className="text-[15px] font-medium text-ink">Settings</h2>
        <button
          onClick={() => setSettingsOpen(false)}
          aria-label="Close settings"
          className="rounded-lg p-1.5 text-ink-3 transition-colors hover:bg-raised hover:text-ink"
        >
          <X size={17} />
        </button>
      </header>

      <div className="px-5 pb-5">
        <h3 className="pt-5 pb-1 text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase">
          Appearance
        </h3>
        <Field label="Theme" hint="Dark suits long reading sessions.">
          <div
            role="radiogroup"
            aria-label="Theme"
            className="flex rounded-lg border border-line-strong p-0.5"
          >
            {(
              [
                ["dark", Moon, "Dark"],
                ["light", Sun, "Light"],
                ["system", Monitor, "System"],
              ] as const
            ).map(([key, Icon, label]) => (
              <button
                key={key}
                role="radio"
                aria-checked={theme === key}
                aria-label={label}
                onClick={() => setTheme(key)}
                className={cx(
                  "flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12px] transition-colors duration-150",
                  theme === key
                    ? "bg-raised text-ink"
                    : "text-ink-3 hover:text-ink",
                )}
              >
                <Icon size={13} />
                {label}
              </button>
            ))}
          </div>
        </Field>

        <h3 className="border-t border-line pt-5 pb-1 text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase">
          Retrieval
        </h3>
        <Field
          label="Passages per answer"
          hint="More context, slower answers."
        >
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={1}
              max={8}
              value={settings.topK}
              aria-label="Passages per answer"
              onChange={(e) => setSettings({ topK: Number(e.target.value) })}
              className="w-32 accent-[var(--color-brass)]"
            />
            <span className="w-4 font-mono text-[12.5px] text-ink">
              {settings.topK}
            </span>
          </div>
        </Field>
        <Field
          label="Similarity floor"
          hint="Drop passages the reranker scores below this."
        >
          <div className="flex items-center gap-3">
            <input
              type="range"
              min={0}
              max={0.95}
              step={0.05}
              value={settings.minScore}
              aria-label="Similarity floor"
              onChange={(e) => setSettings({ minScore: Number(e.target.value) })}
              className="w-32 accent-[var(--color-brass)]"
            />
            <span className="w-8 font-mono text-[12.5px] text-ink">
              {settings.minScore.toFixed(2)}
            </span>
          </div>
        </Field>

        <h3 className="border-t border-line pt-5 pb-1 text-[10.5px] font-semibold tracking-[0.09em] text-ink-3 uppercase">
          Data
        </h3>
        <Field label="Backend" hint={API_BASE}>
          <span className="rounded-full border border-brass/50 px-2.5 py-1 font-mono text-[11px] text-brass">
            live
          </span>
        </Field>
        <Field
          label="Stored conversations"
          hint="Kept in this browser only. Nothing is sent anywhere else."
        >
          <button
            onClick={() => {
              if (confirm("Delete all conversations on this device?")) {
                localStorage.removeItem("hfrag.chat");
                location.reload();
              }
            }}
            className="rounded-lg border border-line-strong px-2.5 py-1.5 text-[12px] text-ink-2 transition-colors hover:border-danger/50 hover:text-danger"
          >
            Clear {conversations.filter((c) => c.messages.length).length}
          </button>
        </Field>
      </div>
    </Modal>
  );
}
