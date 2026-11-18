/**
 * The assistant panel's styles, scoped under .ast-panel (components' style tags are global, see
 * memory: unscoped class names restyle each other), plus the two page-level rules it needs: the
 * docked panel's padding and the highlight pulse (.ast-pulse, on the page's own controls).
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

      /* The NavBar's button: a chip, filled while the panel is open */
      .ast-nav-button {
        display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 0.8rem;
        background: transparent; border: 1px solid var(--oz-rule-strong); color: var(--oz-fg);
        font-size: 0.85rem; font-weight: 500; letter-spacing: 0.01em; line-height: 1.2; cursor: pointer;
        transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
      }
      .ast-nav-button:hover { border-color: var(--oz-mark); color: var(--oz-mark); }
      .ast-nav-button.active { background: var(--oz-fg); border-color: var(--oz-fg); color: var(--oz-bg); }

      .ast-panel {
        --ast-bg: var(--oz-bg); --ast-surface: var(--oz-fill); --ast-border: var(--oz-rule);
        --ast-text: var(--oz-fg); --ast-faint: var(--oz-fg-dim); --ast-user: var(--oz-fill-strong);
        --ast-accent: var(--oz-mark); --ast-warn: var(--oz-warn); --ast-code: var(--oz-fill-strong);
        --ast-strong: var(--oz-fg); --ast-soft: var(--oz-fg-dim); --ast-rail: var(--oz-rule-strong);
        position: fixed; top: 0; right: 0; height: 125vh; z-index: 120;
        display: flex; flex-direction: column; background: var(--ast-bg); color: var(--ast-text);
        border-left: 1px solid var(--ast-border); font-size: 0.9rem; line-height: 1.5;
      }
      .ast-panel--overlay { max-width: calc(125vw - 16px); border-left-color: var(--oz-rule-strong); }

      .ast-panel .ast-resize { position: absolute; left: -3px; top: 0; bottom: 0; width: 6px; cursor: col-resize; z-index: 2; }
      .ast-panel .ast-resize:hover { background: var(--ast-accent); opacity: 0.6; }

      .ast-panel button { font: inherit; color: inherit; }
      .ast-panel svg.lucide { stroke-width: 2.25; }
      .ast-panel .ast-icon {
        display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; flex-shrink: 0;
        background: transparent; border: 1px solid transparent; cursor: pointer; color: var(--ast-strong);
      }
      .ast-panel .ast-icon svg { width: 18px; height: 18px; }
      .ast-panel .ast-icon:hover:not(:disabled) { border-color: var(--oz-rule-strong); color: var(--oz-mark); }
      .ast-panel .ast-icon:disabled { opacity: 0.4; cursor: default; }

      .ast-panel .ast-header { padding: 0.75rem 0.85rem 0.6rem; border-bottom: 1px solid var(--ast-border); display: flex; flex-direction: column; gap: 0.5rem; }
      .ast-panel .ast-header-row { display: flex; align-items: center; gap: 0.35rem; position: relative; }
      .ast-panel .ast-chat-title {
        flex: 1; min-width: 0; display: flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.4rem;
        background: transparent; border: 1px solid transparent; cursor: pointer; font-size: 1rem; font-weight: 500; letter-spacing: -0.015em; text-align: left;
      }
      .ast-panel .ast-chat-title span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-chat-title svg { width: 17px; height: 17px; flex-shrink: 0; color: var(--ast-strong); }
      .ast-panel .ast-chat-title:hover:not(:disabled) { background: var(--ast-surface); }

      .ast-panel .ast-menu {
        position: absolute; top: calc(100% + 4px); left: 0; right: 0; z-index: 5; max-height: 60vh; overflow-y: auto;
        background: var(--ast-bg); border: 1px solid var(--oz-rule-strong); padding: 6px;
      }
      .ast-panel .ast-menu-row { display: flex; align-items: center; gap: 0.15rem; }
      .ast-panel .ast-menu-row.active { background: var(--ast-surface); }
      .ast-panel .ast-menu-row .ast-icon:hover:not(:disabled) { background: transparent; border-color: var(--oz-rule-strong); color: var(--oz-mark); }
      .ast-panel .ast-menu-item {
        display: flex; align-items: center; gap: 0.45rem; width: 100%; padding: 0.4rem 0.5rem; background: transparent;
        border: none; cursor: pointer; text-align: left;
      }
      .ast-panel .ast-menu-item svg { width: 17px; height: 17px; flex-shrink: 0; }
      .ast-panel .ast-menu-item:hover { background: var(--oz-fg); color: var(--oz-bg); }
      .ast-panel .ast-menu-item:hover .ast-menu-price { color: inherit; }
      .ast-panel .ast-menu-title { min-width: 0; justify-content: space-between; }
      .ast-panel .ast-menu-title span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-menu-head { padding: 0.35rem 0.5rem 0.25rem; font-size: 0.78rem; font-weight: 500; color: var(--ast-faint); }
      .ast-panel .ast-menu-foot { padding: 0.45rem 0.5rem 0.25rem; font-size: 0.78rem; color: var(--ast-strong); border-top: 1px solid var(--ast-border); margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
      .ast-panel .ast-menu-check { display: inline-block; width: 17px; height: 17px; flex-shrink: 0; }
      .ast-panel .ast-menu-item .lucide-check { color: var(--oz-mark); }
      .ast-panel .ast-menu-item:hover .lucide-check { color: inherit; }
      .ast-panel .ast-menu--up { top: auto; bottom: calc(100% + 8px); left: 0; right: auto; width: 210px; }
      .ast-panel .ast-menu-label { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-menu-price { font-size: 0.78rem; color: var(--ast-faint); font-variant-numeric: tabular-nums; white-space: nowrap; }
      .ast-panel .ast-bar-control { position: relative; flex-shrink: 0; }
      .ast-panel .ast-bar-button {
        display: inline-flex; align-items: center; gap: 0.3rem; height: 28px; padding: 0 0.6rem; cursor: pointer;
        background: transparent; border: 1px solid var(--oz-rule-strong); color: var(--ast-text); font-size: 0.8rem;
        font-weight: 500; font-variant-numeric: tabular-nums; white-space: nowrap; transition: border-color 0.2s ease, color 0.2s ease;
      }
      .ast-panel .ast-bar-button svg { width: 16px; height: 16px; flex-shrink: 0; color: var(--ast-strong); margin-right: -0.2rem; }
      .ast-panel .ast-bar-button:hover:not(:disabled) { border-color: var(--oz-mark); color: var(--oz-mark); }
      .ast-panel .ast-bar-button:hover:not(:disabled) svg { color: inherit; }
      .ast-panel .ast-bar-button:disabled { opacity: 0.4; cursor: default; }
      .ast-panel .ast-mode { min-width: 5.6rem; justify-content: center; }
      .ast-panel .ast-mode svg { margin: 0 0 0 -0.1rem; }
      .ast-panel .ast-bar-icon {
        display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; padding: 0; cursor: pointer;
        background: transparent; border: 1px solid var(--oz-rule-strong); color: var(--ast-strong); transition: border-color 0.2s ease, color 0.2s ease;
      }
      .ast-panel .ast-bar-icon svg { width: 18px; height: 18px; }
      .ast-panel .ast-bar-icon:hover, .ast-panel .ast-bar-icon[aria-expanded="true"] { border-color: var(--oz-mark); color: var(--oz-mark); }
      .ast-panel .ast-bar-end { margin-left: auto; display: flex; align-items: center; gap: 0.45rem; }
      .ast-panel .ast-settings { width: 220px; }
      .ast-panel .ast-budget { display: flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.5rem 0.4rem; }
      .ast-panel .ast-budget-input {
        flex: 1; min-width: 0; display: flex; align-items: center; gap: 0.25rem; height: 28px; padding: 0 0.45rem;
        background: transparent; border: 1px solid var(--oz-rule-strong); font-variant-numeric: tabular-nums;
      }
      .ast-panel .ast-budget-input:focus-within { border-color: var(--ast-strong); }
      .ast-panel .ast-budget-input input {
        flex: 1; min-width: 0; width: 100%; background: transparent; border: none; outline: none; color: var(--ast-text);
        font: inherit; font-variant-numeric: tabular-nums; -moz-appearance: textfield; appearance: textfield;
      }
      .ast-panel .ast-budget-input input::-webkit-outer-spin-button, .ast-panel .ast-budget-input input::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
      .ast-panel .ast-budget-input input::placeholder { color: var(--ast-faint); }
      .ast-panel .ast-budget-set {
        height: 28px; padding: 0 0.7rem; cursor: pointer; font-size: 0.8rem; font-weight: 500;
        background: var(--ast-text); color: var(--ast-bg) !important; border: none; transition: background 0.25s ease, color 0.25s ease;
      }
      .ast-panel .ast-budget-set:hover:not(:disabled) { background: var(--oz-accent); color: var(--oz-on-accent) !important; }
      .ast-panel .ast-budget-set:disabled { opacity: 0.35; cursor: default; }
      .ast-panel .ast-menu-foot.ast-warn { color: var(--ast-warn); }
      .ast-panel .ast-rename {
        flex: 1; min-width: 0; padding: 0.3rem 0.45rem; background: transparent; color: var(--ast-text);
        border: 1px solid var(--oz-fg); font: inherit; outline: none;
      }

      .ast-panel .ast-body { flex: 1; overflow-y: auto; padding: 0.85rem; display: flex; flex-direction: column; gap: 1.1rem; }
      .ast-panel .ast-empty { margin: auto 0; display: flex; flex-direction: column; align-items: stretch; gap: 0.45rem; }
      /* The homepage's pictogram of the assistant */
      .ast-panel .ast-empty > .ozh-glyph { height: 56px; margin: 0 0 1.5rem; align-self: flex-start; }
      .ast-panel .ast-empty { gap: 0; }
      /* What to ask: a ruled list, as the homepage lists its instruments */
      .ast-panel .ast-suggestion {
        display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; padding: 0.8rem 0; text-align: left;
        cursor: pointer; background: transparent; border: none; border-top: 1px solid var(--ast-border); font-weight: 500;
        transition: color 0.2s ease;
      }
      .ast-panel .ast-suggestion:last-of-type { border-bottom: 1px solid var(--ast-border); }
      .ast-panel .ast-suggestion::after { content: '→'; flex: none; color: var(--ast-faint); transition: transform 0.35s cubic-bezier(0.16, 0.8, 0.2, 1), color 0.2s ease; }
      .ast-panel .ast-suggestion:hover { color: var(--ast-accent); }
      .ast-panel .ast-suggestion:hover::after { color: inherit; transform: translateX(4px); }
      .ast-panel .ast-empty .ast-note { margin-top: 0.9rem; color: var(--ast-faint); }
      .ast-panel .ast-note { font-size: 0.8rem; }

      .ast-panel .ast-turn { display: flex; flex-direction: column; gap: 0.5rem; }
      .ast-panel .ast-user {
        align-self: flex-end; max-width: 88%; padding: 0.5rem 0.7rem; background: var(--ast-user);
        white-space: pre-wrap; overflow-wrap: anywhere;
      }
      .ast-panel .ast-steps {
        display: flex; flex-direction: column; gap: 0.05rem; padding: 0.1rem 0 0.1rem 0.6rem;
        border-left: 1px solid var(--ast-rail); font-size: 0.82rem;
      }
      .ast-panel .ast-step-row { display: flex; align-items: center; gap: 0.4rem; min-width: 0; }
      .ast-panel .ast-step-main {
        flex: 1; min-width: 0; display: flex; align-items: center; gap: 0.45rem; padding: 0.2rem 0.3rem; margin-left: -0.3rem;
        background: transparent; border: none; text-align: left;
      }
      .ast-panel button.ast-step-main { cursor: pointer; }
      .ast-panel button.ast-step-main:hover { background: var(--ast-surface); }
      .ast-panel .ast-step-main > svg { width: 16px; height: 16px; flex-shrink: 0; color: var(--ast-strong); }
      .ast-panel .ast-step-text { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-step-says { font-weight: 500; color: var(--ast-strong); }
      .ast-panel .ast-step-arg { color: var(--ast-soft); }
      .ast-panel .ast-step-end { margin-left: auto; padding-left: 0.4rem; flex-shrink: 0; font-size: 0.75rem; color: var(--ast-soft); font-variant-numeric: tabular-nums; }
      .ast-panel .ast-step-main > .ast-step-chevron { width: 15px; height: 15px; margin-left: auto; transition: transform 0.15s; }
      .ast-panel .ast-step-end + .ast-step-chevron { margin-left: 0; }
      .ast-panel .ast-step-main > .ast-step-chevron.open { transform: rotate(90deg); }
      .ast-panel .ast-step-open {
        flex-shrink: 0; padding: 0.05rem 0.5rem; cursor: pointer; font-size: 0.75rem; font-weight: 500; background: transparent;
        border: 1px solid var(--oz-rule-strong); transition: border-color 0.2s ease, color 0.2s ease;
      }
      .ast-panel .ast-step-open:hover { border-color: var(--ast-accent); color: var(--ast-accent); }
      .ast-panel .ast-step--error .ast-step-main > svg, .ast-panel .ast-step--error .ast-step-end { color: var(--ast-warn); }
      .ast-panel .ast-step--skipped { opacity: 0.6; }
      .ast-panel .ast-step-details {
        margin: 0.15rem 0 0.3rem 1.45rem; padding: 0.45rem 0.6rem; display: flex; flex-direction: column; gap: 0.3rem;
        background: transparent; border: 1px solid var(--ast-border);
      }
      .ast-panel .ast-step-title { font-weight: 500; }

      .ast-panel .ast-approval { border: 1px solid var(--ast-accent); background: transparent; display: flex; flex-direction: column; }
      .ast-panel .ast-approval-notice { padding: 0.45rem 0.7rem; font-size: 0.75rem; color: var(--ast-warn); border-bottom: 1px solid var(--ast-border); }
      .ast-panel .ast-approval-run { padding: 0.6rem 0.7rem; border-bottom: 1px solid var(--ast-border); display: flex; flex-direction: column; gap: 0.35rem; }
      .ast-panel .ast-approval-run--off { opacity: 0.5; }
      .ast-panel .ast-approval-run--call { padding: 0.45rem 0.7rem; }
      .ast-panel .ast-approval-call { min-width: 0; font-weight: 400; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-approval-price.ast-approval-free { font-weight: 400; color: var(--ast-soft); font-size: 0.8rem; }
      .ast-panel .ast-approval-head { display: flex; align-items: baseline; gap: 0.5rem; }
      .ast-panel .ast-approval-title { flex: 1; min-width: 0; font-weight: 500; display: flex; align-items: center; gap: 0.5rem; }
      .ast-panel .ast-approval-title input { margin: 0; }
      .ast-panel .ast-approval-price { font-variant-numeric: tabular-nums; font-weight: 500; }
      .ast-panel .ast-approval-details { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 0.15rem 0.6rem; font-size: 0.78rem; }
      .ast-panel .ast-approval-details > div { display: contents; }
      .ast-panel .ast-approval-details dt { color: var(--ast-faint); }
      .ast-panel .ast-approval-details dd { margin: 0; overflow-wrap: anywhere; }
      .ast-panel .ast-approval-note { font-size: 0.72rem; color: var(--ast-faint); }
      .ast-panel .ast-approval-footer { display: flex; align-items: center; gap: 0.6rem; padding: 0.55rem 0.7rem; font-size: 0.78rem; flex-wrap: wrap; }
      .ast-panel .ast-approval-footer > span { font-variant-numeric: tabular-nums; }
      .ast-panel .ast-approval-actions { margin-left: auto; display: flex; gap: 0.4rem; }
      .ast-panel .ast-approval-skip, .ast-panel .ast-approval-go {
        display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.35rem 0.8rem; cursor: pointer; font-size: 0.8rem; font-weight: 500;
        border: 1px solid var(--oz-rule-strong); background: transparent; transition: background 0.25s ease, border-color 0.25s ease, color 0.25s ease;
      }
      .ast-panel .ast-approval-skip:hover:not(:disabled) { border-color: var(--ast-accent); color: var(--ast-accent); }
      .ast-panel .ast-approval-go { background: var(--ast-text); color: var(--ast-bg) !important; border-color: var(--ast-text); }
      .ast-panel .ast-approval-go:hover:not(:disabled) { background: var(--oz-accent); border-color: var(--oz-accent); color: var(--oz-on-accent) !important; }
      .ast-panel .ast-approval-go svg { width: 15px; height: 15px; }
      .ast-panel .ast-approval-skip:disabled, .ast-panel .ast-approval-go:disabled { opacity: 0.4; cursor: default; }
      .ast-panel .ast-meta { display: flex; align-items: center; gap: 0.6rem; font-size: 0.75rem; color: var(--ast-faint); min-height: 1rem; }
      .ast-panel .ast-cost { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--ast-faint); font-size: 0.72rem; }
      .ast-panel .ast-warn { color: var(--ast-warn); }
      .ast-panel .ast-link { background: none; border: none; border-bottom: 1px solid var(--oz-fg-faint); padding: 0 0 1px; color: var(--ast-text); cursor: pointer; font-size: inherit; font-weight: 500; transition: color 0.2s ease, border-color 0.2s ease; }
      .ast-panel .ast-link:hover:not(:disabled) { color: var(--ast-accent); border-color: var(--ast-accent); }
      .ast-panel .ast-link:disabled { opacity: 0.5; cursor: default; }
      @keyframes ast-spin { to { transform: rotate(360deg); } }
      .ast-panel .ast-spin { animation: ast-spin 0.9s linear infinite; }
      .ast-panel .ast-working { display: inline-flex; align-items: center; gap: 0.5rem; }
      .ast-panel .ast-matrix { display: inline-grid; grid-template-columns: repeat(3, 3.75px); grid-auto-rows: 3.75px; gap: 1.25px; flex: none; }
      .ast-panel .ast-matrix i { background: var(--ast-text); opacity: 0.15; animation: ast-matrix-cell 1.1s ease-in-out infinite; }
      @keyframes ast-matrix-cell { 0%, 100% { opacity: 0.15; } 40% { opacity: 1; } }
      @media (prefers-reduced-motion: reduce) { .ast-panel .ast-matrix i { animation: none; opacity: 0.6; } }

      .ast-panel .ast-markdown { overflow-wrap: anywhere; }
      .ast-panel .ast-markdown > :first-child { margin-top: 0; }
      .ast-panel .ast-markdown > :last-child { margin-bottom: 0; }
      .ast-panel .ast-markdown p, .ast-panel .ast-markdown ul, .ast-panel .ast-markdown ol, .ast-panel .ast-markdown table,
      .ast-panel .ast-markdown pre { margin: 0 0 0.6rem; }
      .ast-panel .ast-markdown ul, .ast-panel .ast-markdown ol { padding-left: 1.2rem; }
      .ast-panel .ast-markdown ul { list-style: disc; }
      .ast-panel .ast-markdown ol { list-style: decimal; }
      .ast-panel .ast-markdown li { margin: 0.15rem 0; }
      .ast-panel .ast-markdown h1, .ast-panel .ast-markdown h2, .ast-panel .ast-markdown h3, .ast-panel .ast-markdown h4 {
        font-size: 0.95rem; font-weight: 500; letter-spacing: -0.015em; margin: 0.8rem 0 0.35rem;
      }
      .ast-panel .ast-markdown code { font-family: var(--oz-mono); font-size: 0.8em; background: var(--ast-code); padding: 0.05rem 0.3rem; }
      .ast-panel .ast-markdown pre { background: var(--ast-code); padding: 0.55rem 0.65rem; overflow-x: auto; }
      .ast-panel .ast-markdown pre code { background: none; padding: 0; }
      .ast-panel .ast-markdown a { color: var(--ast-text); text-decoration: underline; text-decoration-color: var(--oz-fg-faint); text-underline-offset: 3px; transition: color 0.2s ease; }
      .ast-panel .ast-markdown a:hover { color: var(--ast-accent); text-decoration-color: currentColor; }
      .ast-panel .ast-markdown table { border-collapse: collapse; font-size: 0.8rem; display: block; overflow-x: auto; }
      .ast-panel .ast-markdown th, .ast-panel .ast-markdown td { border: 1px solid var(--ast-border); padding: 0.25rem 0.5rem; text-align: left; }
      .ast-panel .ast-markdown th { font-weight: 500; color: var(--ast-faint); }
      .ast-panel .ast-markdown blockquote { border-left: 2px solid var(--ast-border); padding-left: 0.6rem; color: var(--ast-faint); margin: 0 0 0.6rem; }
      .ast-panel .ast-feature a { text-decoration: none; }
      .ast-panel .ast-feature svg { display: inline-block; vertical-align: -1px; margin-left: 2px; }
      .ast-panel .ast-page-chip {
        display: inline; padding: 0.05rem 0.35rem; border: 1px solid var(--oz-rule-strong); background: transparent;
        color: var(--ast-text); cursor: pointer; font-size: 0.85em; font-weight: 500; transition: border-color 0.2s ease, color 0.2s ease;
      }
      .ast-panel .ast-page-chip:hover { border-color: var(--ast-accent); color: var(--ast-accent); }

      .ast-panel .ast-footer { border-top: 1px solid var(--ast-border); padding: 0.65rem 0.85rem 0.75rem; display: flex; flex-direction: column; gap: 0.4rem; }
      .ast-panel .ast-error { display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; color: var(--ast-warn); }
      .ast-panel .ast-error span { flex: 1; }
      .ast-panel .ast-composer {
        display: flex; flex-direction: column; background: transparent; border: 1px solid var(--oz-rule-strong);
        cursor: text; transition: border-color 0.2s ease;
      }
      .ast-panel .ast-composer:focus-within { border-color: var(--ast-strong); }
      .ast-panel .ast-composer textarea {
        display: block; width: 100%; resize: none; min-height: 4rem; max-height: 16rem; overflow-y: auto;
        padding: 0.65rem 0.75rem 0.45rem; font: inherit; background: transparent; color: var(--ast-text); border: none; outline: none;
      }
      .ast-panel .ast-composer textarea::placeholder { color: var(--ast-faint); }
      .ast-panel .ast-composer-bar {
        display: flex; align-items: center; gap: 0.55rem; padding: 0.35rem 0.4rem; border-top: 1px solid var(--ast-border); cursor: default;
      }
      .ast-panel .ast-send {
        display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; flex-shrink: 0;
        background: var(--ast-strong); color: var(--ast-bg) !important; border: none; cursor: pointer; transition: background 0.25s ease, color 0.25s ease;
      }
      .ast-panel .ast-send:hover:not(:disabled) { background: var(--oz-accent); color: var(--oz-on-accent) !important; }
      .ast-panel .ast-send svg { width: 18px; height: 18px; stroke-width: 2.5; }
      .ast-panel .ast-send:disabled { opacity: 0.35; cursor: default; }
      .ast-panel .ast-footnote {
        display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.15rem 0.75rem; font-size: 0.75rem;
        color: var(--ast-text); font-variant-numeric: tabular-nums;
      }
      .ast-panel .ast-footnote-model { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-footnote-chat { margin-left: auto; text-align: right; }
      .ast-panel .ast-footnote-note { margin-top: -0.25rem; font-size: 0.75rem; color: var(--ast-soft); }
    `}</style>
  )
}
