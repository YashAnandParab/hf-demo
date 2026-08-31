import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type ReactNode,
} from "react";

export const cx = (...v: (string | false | null | undefined)[]) =>
  v.filter(Boolean).join(" ");

/* ---------------------------------------------------------------- tooltip */

interface TipProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  label: string;
  /** Where the bubble sits relative to the trigger. */
  side?: "top" | "bottom";
  active?: boolean;
  tone?: "default" | "danger";
}

/** Icon button with a CSS-only tooltip. `label` doubles as the accessible name,
 *  so an icon button is never unlabelled. */
export function IconButton({
  label,
  side = "top",
  active,
  tone = "default",
  className,
  children,
  ...rest
}: TipProps) {
  return (
    <span className="group/tip relative inline-flex">
      <button
        type="button"
        aria-label={label}
        className={cx(
          "inline-flex h-8 w-8 items-center justify-center rounded-lg transition-colors duration-150",
          "text-ink-3 hover:bg-raised hover:text-ink",
          "disabled:pointer-events-none disabled:opacity-40",
          active && "bg-raised !text-brass",
          tone === "danger" && "hover:!text-danger",
          className,
        )}
        {...rest}
      >
        {children}
      </button>
      <span
        role="tooltip"
        className={cx(
          "pointer-events-none absolute left-1/2 z-50 -translate-x-1/2 whitespace-nowrap rounded-md px-2 py-1",
          "bg-raised text-ink text-[11px] font-medium tracking-wide",
          "border border-line-strong shadow-lg shadow-black/30",
          "opacity-0 transition-opacity duration-150",
          "group-hover/tip:opacity-100 group-focus-within/tip:opacity-100",
          side === "top" ? "bottom-full mb-1.5" : "top-full mt-1.5",
        )}
      >
        {label}
      </span>
    </span>
  );
}

/* ------------------------------------------------------------------ popup */

/** Anything that closes on Escape and on an outside click. */
export function Popover({
  open,
  onClose,
  children,
  className,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    const onDown = (e: MouseEvent) => {
      if (!ref.current?.parentElement?.contains(e.target as Node)) onClose();
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onDown);
    };
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      ref={ref}
      className={cx(
        "pop absolute z-50 rounded-xl border border-line-strong bg-raised",
        "shadow-2xl shadow-black/40",
        className,
      )}
    >
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ modal */

export function Modal({
  open,
  onClose,
  title,
  children,
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
}) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-[100] flex items-end justify-center bg-black/55 p-0 backdrop-blur-[2px] sm:items-center sm:p-6"
      onMouseDown={(e) => e.target === e.currentTarget && onClose()}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className="rise max-h-[88vh] w-full max-w-lg overflow-y-auto rounded-t-2xl border border-line-strong bg-surface shadow-2xl shadow-black/50 scroll-quiet sm:rounded-2xl"
      >
        {children}
      </div>
    </div>
  );
}

/* ----------------------------------------------------------------- toasts */

type Toast = { id: number; text: string };
const ToastCtx = createContext<(text: string) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

export function ToastHost({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const push = (text: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, text }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2600);
  };
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div
        aria-live="polite"
        className="pointer-events-none fixed bottom-5 left-1/2 z-[200] flex -translate-x-1/2 flex-col items-center gap-2"
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            className="rise rounded-full border border-line-strong bg-raised px-3.5 py-1.5 text-[13px] text-ink shadow-xl shadow-black/40"
          >
            {t.text}
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

/* --------------------------------------------------------------- skeleton */

export const Skeleton = ({ className }: { className?: string }) => (
  <div className={cx("shimmer rounded-md", className)} />
);
