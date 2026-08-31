# Design system

Written from the built interface, not ahead of it.

## Thesis

**Colour means "this is sourced."** A RAG client earns trust by showing its
receipts, so the single accent in the product is spent on evidence — citation
markers, retrieved passages, similarity meters — and on the one active
affordance (send, active chat, checked model). Chrome is neutral ink
throughout. The category default this refuses: a violet-gradient assistant
whose sources hide behind a tab.

Mode is **Operate**. Expression never obscures the task, the state, or a
familiar affordance.

## Colour

Warm ink, not blue-black — the use scene is a long evening reading session.
Light is its paper-neutral twin, low chroma, not cream. Strategy: **Restrained**
(neutrals plus one accent).

| Token | Dark | Light | Role |
|---|---|---|---|
| `--color-canvas` | `#12110f` | `#faf9f7` | Page ground |
| `--color-surface` | `#1a1917` | `#ffffff` | Sidebar, composer, cards |
| `--color-raised` | `#24221f` | `#ffffff` | Hover, menus, code blocks |
| `--color-line` | `#2d2a26` | `#e7e3dd` | Hairline rules |
| `--color-line-strong` | `#3d3833` | `#d6d0c7` | Interactive borders |
| `--color-ink` | `#eae7e1` | `#1c1a17` | Body text |
| `--color-ink-2` | `#a8a29a` | `#57524b` | Secondary — 7.8:1 / 7.2:1 |
| `--color-ink-3` | `#857e76` | `#6f6960` | Metadata, placeholder — 5.0:1 / 4.9:1 |
| `--color-brass` | `#d9a441` | `#9a6b12` | Evidence + primary action — 8.6:1 / 4.6:1 |
| `--color-danger` | `#e0715f` | `#b03d29` | Errors only |

Every text token clears 4.5:1 against its own ground in both themes. Themes
are token swaps under `html.light`; no colour is defined only inside a
conditional block.

## Type

System sans for UI (`ui-sans-serif, system-ui`) — an Operate surface is well
served by a workhorse stack, and no webfont blocks first paint. Mono
(`ui-monospace`) is reserved for real data: similarity scores, chunk ids, page
numbers, capability tags, code. Never as a "technical" costume.

Answer prose: 15.5px / 1.72, measure capped at 46rem (~68ch). Section labels
are 10.5px uppercase at `0.09em`. Headings balance (`text-wrap: balance`) and
carry more space above than below.

## Form

- Radii 8 / 10 / 12 / 16px. Nothing is a pill except counters and the avatar.
- **Rules, not cards.** Sources, metadata and settings are separated by 1px
  hairlines. The active conversation is marked by a 1px brass rule, not a
  filled block.
- Shadows carry offset and blur (`0 12px 28px -14px`), never a zero-offset
  halo. Only the composer and floating layers are elevated.
- Icons are Lucide at a single stroke weight. No emoji anywhere in the UI.

## Motion

150–250ms, `cubic-bezier(0.22, 1, 0.36, 1)`. One authored moment: content
rises 6px into place as it arrives (`.rise`), staggered only on the starter
questions. The streaming caret is a typesetter's brass mark, not a blinking
terminal block. Everything collapses under `prefers-reduced-motion`.

## Layout

| Breakpoint | Sidebar | Source panel |
|---|---|---|
| `< 768px` | Drawer over a scrim | Full-width drawer |
| `768–1024px` | Resident, collapsible | Drawer |
| `≥ 1024px` | Resident | Resident 23rem column |

Chat column is `max-w-[46rem]`, centred, and the composer shares that measure
so the two align. No horizontal page scroll at any width; wide content (tables,
code) scrolls inside its own container.

## States

Every async path has one: model skeletons, a retrieval indicator before the
first token, per-file upload progress, an inline error with a Try again
action, and distinct empty states for no conversations, no search match, and
no archive. Errors name the problem and the recovery; raw statuses stay in the
console.

## Reserved

`--color-brass` is not a decoration. Before using it on a new element, ask
whether that element is evidence or the single primary action. If neither, it
gets ink.
