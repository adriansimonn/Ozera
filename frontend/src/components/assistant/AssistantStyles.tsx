/**
 * The assistant panel's styles, scoped under .ast-panel (components' style tags are global, see
 * memory: unscoped class names restyle each other), plus the two page-level rules it needs: the
 * docked panel's padding and the highlight pulse (.ast-pulse, on the page's own controls).
 *
 * The panel is drawn like the pages: colors from the app's custom properties, buttons, links and
 * checkboxes from styles/app.css (.oz-btn, .oz-link, .oz-icon-btn, .oz-caret), text controls
 * instead of icons, hairline rules instead of boxes and bubbles, and a square cell for status.
 */

export function AssistantStyles() {
  return (
    <style>{`
      body[data-assistant="docked"] .content-container { padding-right: var(--assist-width); }
      body[data-assistant="docked"] .oz-nav { right: var(--assist-width); }

      /* The control the assistant points at: ringed in the accent, three times */
      @keyframes ast-pulse-ring {
        0%, 100% { box-shadow: 0 0 0 0 transparent; }
        50% { box-shadow: 0 0 0 3px var(--oz-mark); }
      }
      .ast-pulse { animation: ast-pulse-ring 0.85s ease-in-out 3; }

      /* The NavBar's button: a chip (as tall as .oz-chip), filled while the panel is open. Its
         label is trimmed to its capitals, top to baseline, so it's those that sit on the chip's
         centre: a line box's own centre is a rounded pixel or so lower at this size. */
      .ast-nav-button {
        box-sizing: content-box; display: flex; align-items: center; height: calc(1rem + 1.2 * 0.85rem); padding: 0 0.8rem;
        background: transparent; border: 1px solid var(--oz-rule-strong); color: var(--oz-fg);
        font-size: 0.85rem; font-weight: 500; letter-spacing: 0.01em; line-height: 1.2; cursor: pointer;
        transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
      }
      .ast-nav-button span { text-box: trim-both cap alphabetic; }
      .ast-nav-button:hover { border-color: var(--oz-mark); color: var(--oz-mark); }
      .ast-nav-button.active { background: var(--oz-fg); border-color: var(--oz-fg); color: var(--oz-bg); }

      /* ---- The panel: a column ruled off from the page, its head level with the nav ---- */

      .ast-panel {
        position: fixed; top: 0; right: 0; z-index: 120; height: 125vh;
        display: flex; flex-direction: column;
        background: var(--oz-bg); border-left: 1px solid var(--oz-rule);
        color: var(--oz-fg); font-size: 0.9rem; line-height: 1.5;
      }
      .ast-panel--overlay { max-width: calc(125vw - 16px); border-left-color: var(--oz-rule-strong); }

      .ast-panel .ast-resize { position: absolute; left: -3px; top: 0; bottom: 0; z-index: 6; width: 6px; cursor: col-resize; }
      .ast-panel .ast-resize:hover { background: var(--oz-mark); opacity: 0.6; }

      .ast-panel .ast-warn { color: var(--oz-warn); }
      /* What something cost */
      .ast-panel .ast-cost { flex: none; color: var(--oz-fg-faint); font-family: var(--oz-mono); font-size: 0.72rem; font-variant-numeric: tabular-nums; }

      /* ---- Head: the chat's name (which opens the chats), and what to do with it ---- */

      .ast-panel .ast-header {
        position: relative; flex: none; display: flex; align-items: center; gap: 0.95rem;
        height: var(--oz-nav-h); padding: 0 calc(var(--oz-gutter) - 0.5rem) 0 var(--oz-gutter);
        border-bottom: 1px solid var(--oz-rule);
      }
      .ast-panel .ast-chat-title {
        flex: 1; min-width: 0; align-self: stretch; display: flex; align-items: center; gap: 0.55rem;
        font-size: 1.0625rem; font-weight: 500; letter-spacing: -0.02em; text-align: left; cursor: pointer;
        transition: color 0.2s ease;
      }
      .ast-panel .ast-chat-title span { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-chat-title:hover:not(:disabled), .ast-panel .ast-chat-title[aria-expanded="true"] { color: var(--oz-mark); }
      .ast-panel .ast-chat-title:disabled { cursor: default; }
      .ast-panel .ast-head-link {
        flex: none; color: var(--oz-fg-dim); font-size: 0.85rem; font-weight: 500; letter-spacing: 0.01em; cursor: pointer;
        transition: color 0.2s ease;
      }
      .ast-panel .ast-head-link:hover:not(:disabled) { color: var(--oz-mark); }
      .ast-panel .ast-head-link:disabled { color: var(--oz-fg-faint); cursor: default; }
      .ast-panel .ast-header .oz-icon-btn { margin-left: -0.35rem; }

      /* ---- Menus ---- */

      .ast-panel .ast-menu { position: absolute; z-index: 5; overflow-y: auto; background: var(--oz-bg); animation: oz-menu-in 0.22s var(--oz-ease); }

      /* The chats: a ruled list that drops from under the head; the open one is marked on its rule */
      .ast-panel .ast-chats {
        top: calc(100% + 1px); left: 0; right: 0; max-height: 60vh; padding: 0 var(--oz-gutter) 0.4rem;
        border-bottom: 1px solid var(--oz-fg);
      }
      .ast-panel .ast-menu-none { padding: 0.9rem 0 0.6rem; color: var(--oz-fg-dim); font-size: 0.85rem; }
      .ast-panel .ast-chat-row { position: relative; display: flex; align-items: baseline; gap: 0.85rem; color: var(--oz-fg-dim); transition: color 0.2s ease; }
      .ast-panel .ast-chat-row + .ast-chat-row { border-top: 1px solid var(--oz-rule); }
      .ast-panel .ast-chat-row:hover, .ast-panel .ast-chat-row:focus-within, .ast-panel .ast-chat-row.active { color: var(--oz-fg); }
      .ast-panel .ast-chat-row.active::after { content: ''; position: absolute; z-index: 1; left: 0; right: 0; bottom: -1px; height: 1px; background: var(--oz-mark); }
      .ast-panel .ast-chat-open {
        flex: 1; min-width: 0; display: flex; align-items: baseline; gap: 0.85rem; padding: 0.75rem 0;
        font-weight: 500; text-align: left; cursor: pointer;
      }
      .ast-panel .ast-chat-name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      /* Under the pointer a chat shows what can be done with it, in place of what it cost */
      .ast-panel .ast-row-actions { display: none; flex: none; align-items: baseline; gap: 0.85rem; }
      .ast-panel .ast-chat-row:hover .ast-row-actions, .ast-panel .ast-chat-row:focus-within .ast-row-actions { display: inline-flex; }
      .ast-panel .ast-chat-row:hover .ast-chat-open .ast-cost, .ast-panel .ast-chat-row:focus-within .ast-chat-open .ast-cost { display: none; }
      .ast-panel .ast-row-action { flex: none; color: var(--oz-fg-dim); font-size: 0.8rem; font-weight: 500; cursor: pointer; transition: color 0.2s ease; }
      .ast-panel .ast-row-action:hover { color: var(--oz-mark); }
      .ast-panel .ast-rename {
        flex: 1; min-width: 0; margin: 0.45rem 0; padding: 0.3rem 0.55rem; background: transparent; border: 1px solid var(--oz-fg);
        border-radius: 0; outline: none; color: var(--oz-fg); font: inherit; font-weight: 500;
      }

      /* The model and the budget: boxes that open up from the bar; the chosen option carries a lit cell */
      .ast-panel .ast-menu--up {
        left: 0; bottom: calc(100% + 0.6rem); width: 15rem; max-width: 100%; max-height: 60vh; padding: 6px;
        border: 1px solid var(--oz-fg); animation-name: ast-menu-up;
      }
      @keyframes ast-menu-up { from { opacity: 0; transform: translateY(6px); } }
      .ast-panel .ast-menu-head { padding: 0.45rem 0.75rem 0.3rem; color: var(--oz-fg-dim); font-size: 0.78rem; font-weight: 500; }
      .ast-panel .ast-menu-item {
        position: relative; display: flex; align-items: baseline; gap: 0.75rem; width: 100%; padding: 0.5rem 0.75rem 0.5rem 1.6rem;
        font-weight: 500; text-align: left; cursor: pointer; transition: background 0.12s ease, color 0.12s ease;
      }
      .ast-panel .ast-menu-item:hover { background: var(--oz-fg); color: var(--oz-bg); }
      .ast-panel .ast-menu-item[aria-checked="true"]::before {
        content: ''; position: absolute; top: 50%; left: 0.65rem; width: 6px; height: 6px; margin-top: -3px; background: var(--oz-mark);
      }
      .ast-panel .ast-menu-item[aria-checked="true"]:hover::before { background: currentColor; }
      .ast-panel .ast-menu-label { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-menu-price { flex: none; color: var(--oz-fg-dim); font-family: var(--oz-mono); font-size: 0.72rem; font-variant-numeric: tabular-nums; }
      .ast-panel .ast-menu-item:hover .ast-menu-price { color: inherit; }
      .ast-panel .ast-menu-foot {
        margin-top: 6px; padding: 0.55rem 0.75rem 0.35rem; border-top: 1px solid var(--oz-rule);
        color: var(--oz-fg-dim); font-size: 0.78rem; font-variant-numeric: tabular-nums;
      }
      .ast-panel .ast-menu-foot.ast-warn { color: var(--oz-warn); }
      .ast-panel .ast-budget { display: flex; align-items: stretch; gap: 6px; padding: 0.35rem 0.75rem 0.5rem 1.6rem; }
      .ast-panel .ast-budget-input {
        flex: 1; min-width: 0; display: flex; align-items: center; gap: 0.3rem; padding: 0 0.6rem;
        border: 1px solid var(--oz-rule-strong); font-variant-numeric: tabular-nums; transition: border-color 0.2s ease;
      }
      .ast-panel .ast-budget-input:focus-within { border-color: var(--oz-fg); }
      .ast-panel .ast-budget-input > span { color: var(--oz-fg-dim); }
      .ast-panel .ast-budget-input input {
        flex: 1; min-width: 0; width: 100%; background: transparent; border: none; outline: none; color: var(--oz-fg);
        font: inherit; font-variant-numeric: tabular-nums;
      }
      .ast-panel .ast-budget-input input::placeholder { color: var(--oz-fg-faint); }

      /* ---- The chat: turns ruled off from each other, each headed by what was asked ---- */

      .ast-panel .ast-body { flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; padding: 0 var(--oz-gutter); }
      .ast-panel .ast-note { margin: 1.1rem 0 0; font-size: 0.85rem; }

      .ast-panel .ast-empty { margin: auto 0; padding: 2rem 0; display: flex; flex-direction: column; align-items: stretch; }
      /* The homepage's pictogram of the assistant */
      .ast-panel .ast-empty > .ozh-glyph { height: 56px; margin: 0 0 1.5rem; align-self: flex-start; }
      /* What to ask: a ruled list, as the homepage lists its instruments */
      .ast-panel .ast-suggestion {
        display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; padding: 0.8rem 0; text-align: left;
        cursor: pointer; border-top: 1px solid var(--oz-rule); font-weight: 500; transition: color 0.2s ease;
      }
      .ast-panel .ast-suggestion:last-of-type { border-bottom: 1px solid var(--oz-rule); }
      .ast-panel .ast-suggestion::after { content: '→'; flex: none; color: var(--oz-fg-dim); transition: transform 0.35s var(--oz-ease), color 0.2s ease; }
      .ast-panel .ast-suggestion:hover { color: var(--oz-mark); }
      .ast-panel .ast-suggestion:hover::after { color: inherit; transform: translateX(4px); }

      .ast-panel .ast-turn { display: flex; flex-direction: column; gap: 0.75rem; padding: 1.4rem 0 1.1rem; }
      .ast-panel .ast-turn + .ast-turn { border-top: 1px solid var(--oz-rule); }
      .ast-panel .ast-user { font-size: 1.0625rem; font-weight: 500; letter-spacing: -0.02em; line-height: 1.35; white-space: pre-wrap; overflow-wrap: anywhere; }
      .ast-panel .ast-meta { display: flex; align-items: baseline; gap: 0.85rem; min-height: 1rem; color: var(--oz-fg-dim); font-size: 0.8rem; }
      .ast-panel .ast-meta .ast-cost { margin-left: auto; }

      /* Working: nine cells lit in a wave */
      .ast-panel .ast-working { display: inline-flex; align-items: center; gap: 0.6rem; }
      .ast-panel .ast-matrix { display: inline-grid; grid-template-columns: repeat(3, 3.75px); grid-auto-rows: 3.75px; gap: 1.25px; flex: none; }
      .ast-panel .ast-matrix i { background: var(--oz-fg); opacity: 0.15; animation: ast-matrix-cell 1.1s ease-in-out infinite; }
      @keyframes ast-matrix-cell { 0%, 100% { opacity: 0.15; } 40% { opacity: 1; } }

      /* ---- Tool calls: a row each, with a cell for how it went ---- */

      .ast-panel .ast-steps { display: flex; flex-direction: column; font-size: 0.82rem; }
      .ast-panel .ast-step-row { display: flex; align-items: baseline; gap: 0.85rem; min-width: 0; }
      .ast-panel .ast-step-main { flex: 1; min-width: 0; display: flex; align-items: baseline; gap: 0.65rem; padding: 0.24rem 0; text-align: left; }
      .ast-panel button.ast-step-main { cursor: pointer; transition: color 0.2s ease; }
      .ast-panel button.ast-step-main:hover { color: var(--oz-mark); }
      /* Lit when done; pulsing while it runs; hollow while it waits or once it was skipped */
      .ast-panel .ast-cell { flex: none; width: 7px; height: 7px; background: var(--oz-fg); }
      .ast-panel .ast-step--running .ast-cell { background: var(--oz-mark); animation: oz-cell 1s ease-in-out infinite; }
      .ast-panel .ast-step--waiting .ast-cell { background: transparent; box-shadow: inset 0 0 0 1px var(--oz-fg); }
      .ast-panel .ast-step--skipped .ast-cell { background: transparent; box-shadow: inset 0 0 0 1px var(--oz-fg-faint); }
      .ast-panel .ast-step--failed .ast-cell { background: var(--oz-danger); }
      .ast-panel .ast-step-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-step-says { font-weight: 500; }
      .ast-panel .ast-step-arg { color: var(--oz-fg-dim); }
      .ast-panel .ast-step--skipped .ast-step-says { color: var(--oz-fg-dim); }
      .ast-panel .ast-step-end { flex: none; margin-left: auto; color: var(--oz-fg-dim); font-family: var(--oz-mono); font-size: 0.72rem; font-variant-numeric: tabular-nums; }
      .ast-panel .ast-step--failed .ast-step-end { color: var(--oz-danger); }
      .ast-panel .ast-step-main > .oz-caret { align-self: center; margin-left: auto; color: var(--oz-fg-dim); }
      .ast-panel .ast-step-end + .oz-caret { margin-left: 0; }
      .ast-panel .ast-step-open { flex: none; font-size: 0.78rem; }
      /* A run's quote, opened under its row */
      .ast-panel .ast-step-details {
        display: flex; flex-direction: column; gap: 0.45rem; margin: 0.3rem 0 0.55rem calc(7px + 0.65rem); padding: 0.65rem 0 0.7rem;
        border-top: 1px solid var(--oz-rule); border-bottom: 1px solid var(--oz-rule);
      }
      .ast-panel .ast-step-title { font-weight: 500; }

      /* ---- Calls waiting for the user: one box, a row a call, and the two answers ---- */

      .ast-panel .ast-approval { display: flex; flex-direction: column; border: 1px solid var(--oz-mark); }
      .ast-panel .ast-approval-notice { padding: 0.55rem 0.85rem; border-bottom: 1px solid var(--oz-rule); color: var(--oz-warn); font-size: 0.8rem; }
      .ast-panel .ast-approval-run {
        display: flex; flex-direction: column; gap: 0.45rem; padding: 0.8rem 0.85rem; border-bottom: 1px solid var(--oz-rule);
        transition: opacity 0.2s ease;
      }
      .ast-panel .ast-approval-run--call { padding: 0.6rem 0.85rem; }
      .ast-panel .ast-approval-run--off { opacity: 0.45; }
      .ast-panel .ast-approval-head { display: flex; align-items: baseline; gap: 0.75rem; }
      .ast-panel .ast-approval-title { flex: 1; min-width: 0; display: flex; align-items: center; gap: 0.6rem; font-weight: 500; }
      .ast-panel label.ast-approval-title { cursor: pointer; }
      .ast-panel .ast-approval-title input { flex: none; margin: 0; }
      .ast-panel .ast-approval-call { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 400; }
      .ast-panel .ast-approval-price { flex: none; font-family: var(--oz-mono); font-size: 0.8rem; font-variant-numeric: tabular-nums; }
      .ast-panel .ast-approval-free { color: var(--oz-fg-dim); }
      .ast-panel .ast-approval-details { display: grid; grid-template-columns: auto 1fr; gap: 0.2rem 1rem; margin: 0; font-size: 0.8rem; }
      .ast-panel .ast-approval-details > div { display: contents; }
      .ast-panel .ast-approval-details dt { color: var(--oz-fg-dim); }
      .ast-panel .ast-approval-details dd { margin: 0; overflow-wrap: anywhere; }
      .ast-panel .ast-approval-note { color: var(--oz-fg-dim); font-size: 0.76rem; line-height: 1.45; }
      .ast-panel .ast-approval-footer { display: flex; align-items: center; flex-wrap: wrap; gap: 0.6rem 0.85rem; padding: 0.7rem 0.85rem; }
      .ast-panel .ast-approval-footer > span { color: var(--oz-fg-dim); font-size: 0.78rem; font-variant-numeric: tabular-nums; }
      .ast-panel .ast-approval-footer > span.ast-warn { color: var(--oz-warn); }
      .ast-panel .ast-approval-footer .oz-link { font-size: 0.8rem; }
      .ast-panel .ast-approval-actions { margin-left: auto; display: flex; gap: 6px; }

      /* ---- A reply's text ---- */

      .ast-panel .ast-markdown { overflow-wrap: anywhere; }
      .ast-panel .ast-markdown > :first-child { margin-top: 0; }
      .ast-panel .ast-markdown > :last-child { margin-bottom: 0; }
      .ast-panel .ast-markdown p, .ast-panel .ast-markdown ul, .ast-panel .ast-markdown ol, .ast-panel .ast-markdown table,
      .ast-panel .ast-markdown pre, .ast-panel .ast-markdown blockquote { margin: 0 0 0.7rem; }
      .ast-panel .ast-markdown ul, .ast-panel .ast-markdown ol { padding-left: 1.15rem; }
      .ast-panel .ast-markdown ul { list-style: square; }
      .ast-panel .ast-markdown ol { list-style: decimal; }
      .ast-panel .ast-markdown li { margin: 0.2rem 0; }
      .ast-panel .ast-markdown li::marker { color: var(--oz-fg-dim); }
      .ast-panel .ast-markdown strong { font-weight: 600; }
      .ast-panel .ast-markdown h1, .ast-panel .ast-markdown h2, .ast-panel .ast-markdown h3, .ast-panel .ast-markdown h4 {
        margin: 1rem 0 0.4rem; font-size: 1rem; font-weight: 500; letter-spacing: -0.02em; line-height: 1.3;
      }
      .ast-panel .ast-markdown hr { margin: 0.9rem 0; border: none; border-top: 1px solid var(--oz-rule); }
      .ast-panel .ast-markdown code { padding: 0.05rem 0.3rem; background: var(--oz-fill-strong); font-family: var(--oz-mono); font-size: 0.82em; }
      .ast-panel .ast-markdown pre { padding: 0.65rem 0.8rem; overflow-x: auto; border: 1px solid var(--oz-rule); line-height: 1.5; }
      .ast-panel .ast-markdown pre code { padding: 0; background: none; }
      .ast-panel .ast-markdown a {
        color: inherit; text-decoration: underline; text-decoration-color: var(--oz-fg-faint); text-underline-offset: 3px;
        transition: color 0.2s ease, text-decoration-color 0.2s ease;
      }
      .ast-panel .ast-markdown a:hover { color: var(--oz-mark); text-decoration-color: currentColor; }
      /* A table: ruled rows, no boxes */
      .ast-panel .ast-markdown table { display: block; overflow-x: auto; border-collapse: collapse; font-size: 0.82rem; font-variant-numeric: tabular-nums; }
      .ast-panel .ast-markdown th, .ast-panel .ast-markdown td { padding: 0.4rem 1.1rem 0.4rem 0; border-bottom: 1px solid var(--oz-rule); text-align: left; }
      .ast-panel .ast-markdown th { border-bottom-color: var(--oz-rule-strong); color: var(--oz-fg-dim); font-weight: 500; }
      .ast-panel .ast-markdown blockquote { padding-left: 0.8rem; border-left: 1px solid var(--oz-rule-strong); color: var(--oz-fg-dim); }
      .ast-panel .ast-feature a { text-decoration: none; }
      .ast-panel .ast-feature svg { display: inline-block; vertical-align: -1px; margin-left: 2px; }
      .ast-panel .ast-page-chip {
        display: inline; padding: 0.05rem 0.4rem; border: 1px solid var(--oz-rule-strong);
        font-size: 0.85em; font-weight: 500; cursor: pointer; transition: border-color 0.2s ease, color 0.2s ease;
      }
      .ast-panel .ast-page-chip:hover { border-color: var(--oz-mark); color: var(--oz-mark); }

      /* ---- Foot: the message, then how it's sent ---- */

      .ast-panel .ast-footer { flex: none; display: flex; flex-direction: column; gap: 0.7rem; padding: 1rem var(--oz-gutter) 1.1rem; border-top: 1px solid var(--oz-rule); }
      .ast-panel .ast-error {
        display: flex; align-items: center; gap: 0.75rem; padding: 0.3rem 0.3rem 0.3rem 0.85rem;
        border: 1px solid var(--oz-rule); border-left: 2px solid var(--oz-warn); font-size: 0.85rem; line-height: 1.45;
      }
      .ast-panel .ast-error > span { flex: 1; min-width: 0; padding: 0.25rem 0; }
      .ast-panel .ast-error .oz-link { flex: none; }
      /* The message and Send share a box, like a field on a page. Send is a square holding the
         app's arrow; while a reply runs it holds a cell, and stops it. The square, the arrow and
         the cell are whole pixels on screen under the page's 0.8 zoom (26, 12 and 8), with even
         differences, so what's inside lands on the exact centre instead of half a pixel off it. */
      .ast-panel .ast-composer { display: flex; flex-direction: column; border: 1px solid var(--oz-rule-strong); cursor: text; transition: border-color 0.2s ease; }
      .ast-panel .ast-composer:hover:not(:focus-within) { border-color: var(--oz-fg-faint); }
      .ast-panel .ast-composer:focus-within { border-color: var(--oz-fg); }
      .ast-panel .ast-input {
        display: block; width: 100%; min-height: 3.4rem; max-height: 16rem; resize: none; overflow-y: auto;
        padding: 0.7rem 0.85rem 0.3rem; background: transparent; border: none; border-radius: 0; outline: none; box-shadow: none;
        color: var(--oz-fg); font: inherit; line-height: 1.5;
      }
      .ast-panel .ast-input::placeholder { color: var(--oz-fg-faint); }
      .ast-panel .ast-composer-end { display: flex; justify-content: flex-end; padding: 0 6px 6px; }
      .ast-panel .ast-send { width: 32.5px; height: 32.5px; padding: 0; cursor: pointer; }
      .ast-panel .ast-send:disabled { cursor: default; }
      .ast-panel .ast-send .ast-send-arrow {
        width: 15px; height: 15px; fill: none; stroke: currentColor; stroke-width: 1.5; stroke-linecap: square; stroke-linejoin: miter;
        transition: transform 0.35s var(--oz-ease);
      }
      .ast-panel .ast-send:hover:not(:disabled) .ast-send-arrow { transform: translateX(2px); }
      .ast-panel .ast-send-stop { width: 10px; height: 10px; background: currentColor; }

      /* Under it, as the nav reads: the chat's model and budget, and how its tool calls run */
      .ast-panel .ast-composer-bar { position: relative; display: flex; align-items: center; flex-wrap: wrap; gap: 0.3rem 1.25rem; }
      .ast-panel .ast-bar-control { flex: none; }
      .ast-panel .ast-bar-link {
        display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.3rem 0; font-size: 0.85rem; font-weight: 500;
        letter-spacing: 0.01em; white-space: nowrap; cursor: pointer; transition: color 0.2s ease;
      }
      .ast-panel .ast-bar-link:hover:not(:disabled), .ast-panel .ast-bar-link[aria-expanded="true"] { color: var(--oz-mark); }
      .ast-panel .ast-bar-link:disabled { color: var(--oz-fg-faint); cursor: default; }
      /* These open upward, so their carets start the other way up */
      .ast-panel .ast-bar-link > .oz-caret { transform: rotate(180deg); }
      .ast-panel .ast-bar-link[aria-expanded="true"] > .oz-caret { transform: none; }
      /* Manual or auto: the one in force is marked under it, like the page you're on */
      .ast-panel .ast-mode { margin-left: auto; display: flex; align-items: center; gap: 0.95rem; }
      .ast-panel .ast-mode-option {
        position: relative; padding: 0.3rem 0; color: var(--oz-fg-dim); font-size: 0.85rem; font-weight: 500; letter-spacing: 0.01em;
        cursor: pointer; transition: color 0.2s ease;
      }
      .ast-panel .ast-mode-option:hover, .ast-panel .ast-mode-option[aria-checked="true"] { color: var(--oz-fg); }
      .ast-panel .ast-mode-option::after {
        content: ''; position: absolute; inset: auto 0 0 0; height: 2px; background: var(--oz-mark);
        clip-path: inset(0 100% 0 0); transition: clip-path 0.4s var(--oz-ease);
      }
      .ast-panel .ast-mode-option[aria-checked="true"]::after { clip-path: inset(0); }

      .ast-panel .ast-footnote {
        display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.15rem 1rem; color: var(--oz-fg-dim);
        font-size: 0.76rem; font-variant-numeric: tabular-nums;
      }
      .ast-panel .ast-footnote-model { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-footnote-chat { margin-left: auto; text-align: right; }
      .ast-panel .ast-footnote-note { margin-top: -0.35rem; color: var(--oz-fg-dim); font-size: 0.76rem; }

      @media (prefers-reduced-motion: reduce) {
        .ast-panel .ast-matrix i { animation: none; opacity: 0.6; }
        .ast-panel .ast-step--running .ast-cell { animation: none; }
        .ast-panel .ast-menu { animation: none; }
        .ast-panel .ast-mode-option::after { transition: none; }
      }
    `}</style>
  )
}
