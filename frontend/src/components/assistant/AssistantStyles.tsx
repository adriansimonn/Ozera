/**
 * The assistant panel's styles, scoped under .ast-panel (components' style tags are global, see
 * memory: unscoped class names restyle each other), plus the two page-level rules it needs: the
 * docked panel's padding and the highlight pulse (.ast-pulse, on the page's own controls).
 */

export function AssistantStyles() {
  return (
    <style>{`
      body[data-assistant="docked"] .content-container { padding-right: var(--assist-width); }
      body[data-assistant="docked"] .navbar { right: var(--assist-width); }

      @keyframes ast-pulse-ring {
        0%, 100% { box-shadow: 0 0 0 0 rgba(96, 165, 250, 0); }
        50% { box-shadow: 0 0 0 4px rgba(96, 165, 250, 0.65); }
      }
      .ast-pulse { animation: ast-pulse-ring 0.85s ease-in-out 3; border-radius: 4px; }

      .ast-nav-button {
        display: flex; align-items: center; gap: 0.45rem; padding: 0.5rem 0.85rem;
        background: transparent; border: 1px solid rgba(255, 255, 255, 0.12); color: #fff;
        font-size: 0.85rem; font-weight: 500; cursor: pointer; transition: background 0.15s, border-color 0.15s;
      }
      .ast-nav-button svg { width: 18px; height: 18px; stroke-width: 2.25; }
      .ast-nav-button:hover { background: rgba(255, 255, 255, 0.06); }
      .ast-nav-button.active { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.22); }
      [data-bg="light"] .ast-nav-button { color: #1d1d1f; border-color: rgba(0, 0, 0, 0.12); }
      [data-bg="light"] .ast-nav-button:hover { background: rgba(0, 0, 0, 0.04); }
      [data-bg="light"] .ast-nav-button.active { background: rgba(0, 0, 0, 0.07); border-color: rgba(0, 0, 0, 0.2); }

      .ast-panel {
        --ast-bg: #0d0d0e; --ast-surface: rgba(255, 255, 255, 0.04); --ast-border: rgba(255, 255, 255, 0.1);
        --ast-text: #ffffff; --ast-faint: rgba(255, 255, 255, 0.55); --ast-user: rgba(255, 255, 255, 0.07);
        --ast-accent: rgba(96, 165, 250, 0.95); --ast-warn: #fab219; --ast-code: rgba(255, 255, 255, 0.08);
        --ast-strong: #ffffff; --ast-soft: rgba(255, 255, 255, 0.8); --ast-rail: rgba(255, 255, 255, 0.2);
        position: fixed; top: 0; right: 0; height: 125vh; z-index: 120;
        display: flex; flex-direction: column; background: var(--ast-bg); color: var(--ast-text);
        border-left: 1px solid var(--ast-border); font-size: 0.875rem; line-height: 1.5;
      }
      [data-bg="light"] .ast-panel {
        --ast-bg: #ffffff; --ast-surface: rgba(0, 0, 0, 0.03); --ast-border: rgba(0, 0, 0, 0.1);
        --ast-text: #1d1d1f; --ast-faint: rgba(0, 0, 0, 0.55); --ast-user: rgba(0, 0, 0, 0.05);
        --ast-accent: #2563eb; --ast-warn: #b77900; --ast-code: rgba(0, 0, 0, 0.06);
        --ast-strong: #000000; --ast-soft: rgba(0, 0, 0, 0.72); --ast-rail: rgba(0, 0, 0, 0.18);
      }
      [data-interface="glass"]:not([data-bg="light"]) .ast-panel {
        background: rgba(8, 8, 10, 0.82); backdrop-filter: blur(24px) saturate(180%); -webkit-backdrop-filter: blur(24px) saturate(180%);
      }
      [data-bg="light"][data-interface="glass"] .ast-panel {
        background: rgba(255, 255, 255, 0.85); backdrop-filter: blur(24px) saturate(180%); -webkit-backdrop-filter: blur(24px) saturate(180%);
      }
      .ast-panel--overlay { max-width: calc(125vw - 16px); box-shadow: -12px 0 40px rgba(0, 0, 0, 0.45); }

      .ast-panel .ast-resize { position: absolute; left: -3px; top: 0; bottom: 0; width: 6px; cursor: col-resize; z-index: 2; }
      .ast-panel .ast-resize:hover { background: var(--ast-accent); opacity: 0.4; }

      .ast-panel button { font: inherit; color: inherit; }
      .ast-panel svg.lucide { stroke-width: 2.25; }
      .ast-panel .ast-icon {
        display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; flex-shrink: 0;
        background: transparent; border: 1px solid transparent; cursor: pointer; color: var(--ast-strong);
      }
      .ast-panel .ast-icon svg { width: 18px; height: 18px; }
      .ast-panel .ast-icon:hover:not(:disabled) { background: var(--ast-surface); }
      .ast-panel .ast-icon:disabled { opacity: 0.4; cursor: default; }

      .ast-panel .ast-header { padding: 0.75rem 0.85rem 0.6rem; border-bottom: 1px solid var(--ast-border); display: flex; flex-direction: column; gap: 0.5rem; }
      .ast-panel .ast-header-row { display: flex; align-items: center; gap: 0.35rem; position: relative; }
      .ast-panel .ast-chat-title {
        flex: 1; min-width: 0; display: flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.4rem;
        background: transparent; border: 1px solid transparent; cursor: pointer; font-weight: 600; text-align: left;
      }
      .ast-panel .ast-chat-title span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-chat-title svg { width: 17px; height: 17px; flex-shrink: 0; color: var(--ast-strong); }
      .ast-panel .ast-chat-title:hover:not(:disabled) { background: var(--ast-surface); }

      .ast-panel .ast-menu {
        position: absolute; top: calc(100% + 4px); left: 0; right: 0; z-index: 5; max-height: 60vh; overflow-y: auto;
        background: var(--ast-bg); border: 1px solid var(--ast-border); box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35); padding: 0.25rem;
      }
      .ast-panel .ast-menu-row { display: flex; align-items: center; gap: 0.15rem; }
      .ast-panel .ast-menu-row.active { background: var(--ast-surface); }
      .ast-panel .ast-menu-item {
        display: flex; align-items: center; gap: 0.45rem; width: 100%; padding: 0.4rem 0.5rem; background: transparent;
        border: none; cursor: pointer; text-align: left;
      }
      .ast-panel .ast-menu-item svg { width: 17px; height: 17px; flex-shrink: 0; color: var(--ast-strong); }
      .ast-panel .ast-menu-item:hover { background: var(--ast-surface); }
      .ast-panel .ast-menu-title { min-width: 0; justify-content: space-between; }
      .ast-panel .ast-menu-title span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-menu-head { padding: 0.35rem 0.5rem 0.25rem; font-size: 0.72rem; color: var(--ast-faint); }
      .ast-panel .ast-menu-foot { padding: 0.35rem 0.5rem 0.25rem; font-size: 0.72rem; color: var(--ast-faint); border-top: 1px solid var(--ast-border); margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
      .ast-panel .ast-menu-check { display: inline-block; width: 17px; height: 17px; flex-shrink: 0; }
      .ast-panel .ast-menu--up { top: auto; bottom: calc(100% + 8px); left: 0; right: auto; width: 210px; }
      .ast-panel .ast-menu-label { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-menu-price { font-size: 0.72rem; color: var(--ast-faint); font-variant-numeric: tabular-nums; white-space: nowrap; }
      .ast-panel .ast-bar-control { position: relative; flex-shrink: 0; }
      .ast-panel .ast-bar-button {
        display: inline-flex; align-items: center; gap: 0.3rem; height: 28px; padding: 0 0.6rem; cursor: pointer;
        background: transparent; border: 1px solid var(--ast-border); color: var(--ast-text); font-size: 0.78rem;
        font-weight: 500; font-variant-numeric: tabular-nums; white-space: nowrap;
      }
      .ast-panel .ast-bar-button svg { width: 16px; height: 16px; flex-shrink: 0; color: var(--ast-strong); margin-right: -0.2rem; }
      .ast-panel .ast-bar-button:hover:not(:disabled) { background: var(--ast-surface); }
      .ast-panel .ast-bar-button:disabled { opacity: 0.4; cursor: default; }
      .ast-panel .ast-bar-button.on { color: var(--ast-accent); border-color: var(--ast-accent); }
      .ast-panel .ast-rename {
        flex: 1; min-width: 0; padding: 0.3rem 0.45rem; background: var(--ast-surface); color: var(--ast-text);
        border: 1px solid var(--ast-border); font: inherit; outline: none;
      }

      .ast-panel .ast-body { flex: 1; overflow-y: auto; padding: 0.85rem; display: flex; flex-direction: column; gap: 1.1rem; }
      .ast-panel .ast-empty { margin: auto 0; display: flex; flex-direction: column; align-items: stretch; gap: 0.45rem; }
      .ast-panel .ast-empty > svg { width: 30px; height: 30px; color: var(--ast-strong); align-self: center; margin-bottom: 0.4rem; }
      .ast-panel .ast-suggestion {
        padding: 0.55rem 0.7rem; text-align: left; cursor: pointer; background: var(--ast-surface);
        border: 1px solid var(--ast-border); transition: border-color 0.15s;
      }
      .ast-panel .ast-suggestion:hover { border-color: var(--ast-accent); }
      .ast-panel .ast-note { font-size: 0.8rem; }

      .ast-panel .ast-turn { display: flex; flex-direction: column; gap: 0.5rem; }
      .ast-panel .ast-user {
        align-self: flex-end; max-width: 88%; padding: 0.5rem 0.7rem; background: var(--ast-user);
        white-space: pre-wrap; overflow-wrap: anywhere;
      }
      .ast-panel .ast-steps {
        display: flex; flex-direction: column; gap: 0.05rem; padding: 0.1rem 0 0.1rem 0.6rem;
        border-left: 2px solid var(--ast-rail); font-size: 0.8rem;
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
        flex-shrink: 0; padding: 0.05rem 0.5rem; cursor: pointer; font-size: 0.72rem; background: transparent;
        border: 1px solid var(--ast-border);
      }
      .ast-panel .ast-step-open:hover { border-color: var(--ast-accent); }
      .ast-panel .ast-step--error .ast-step-main > svg, .ast-panel .ast-step--error .ast-step-end { color: var(--ast-warn); }
      .ast-panel .ast-step--skipped { opacity: 0.6; }
      .ast-panel .ast-step-details {
        margin: 0.15rem 0 0.3rem 1.45rem; padding: 0.45rem 0.6rem; display: flex; flex-direction: column; gap: 0.3rem;
        background: var(--ast-surface); border: 1px solid var(--ast-border);
      }
      .ast-panel .ast-step-title { font-weight: 600; }

      .ast-panel .ast-approval { border: 1px solid var(--ast-accent); background: var(--ast-surface); display: flex; flex-direction: column; }
      .ast-panel .ast-approval-notice { padding: 0.45rem 0.7rem; font-size: 0.75rem; color: var(--ast-warn); border-bottom: 1px solid var(--ast-border); }
      .ast-panel .ast-approval-run { padding: 0.6rem 0.7rem; border-bottom: 1px solid var(--ast-border); display: flex; flex-direction: column; gap: 0.35rem; }
      .ast-panel .ast-approval-run--off { opacity: 0.5; }
      .ast-panel .ast-approval-head { display: flex; align-items: baseline; gap: 0.5rem; }
      .ast-panel .ast-approval-title { flex: 1; min-width: 0; font-weight: 600; display: flex; align-items: center; gap: 0.4rem; }
      .ast-panel .ast-approval-title input { accent-color: var(--ast-accent); margin: 0; }
      .ast-panel .ast-approval-price { font-variant-numeric: tabular-nums; font-weight: 600; }
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
        border: 1px solid var(--ast-border); background: transparent;
      }
      .ast-panel .ast-approval-skip:hover:not(:disabled) { background: var(--ast-surface); }
      .ast-panel .ast-approval-go { background: var(--ast-text); color: var(--ast-bg) !important; border-color: transparent; }
      .ast-panel .ast-approval-go svg { width: 15px; height: 15px; }
      .ast-panel .ast-approval-skip:disabled, .ast-panel .ast-approval-go:disabled { opacity: 0.4; cursor: default; }
      .ast-panel .ast-meta { display: flex; align-items: center; gap: 0.6rem; font-size: 0.75rem; color: var(--ast-faint); min-height: 1rem; }
      .ast-panel .ast-cost { margin-left: auto; font-variant-numeric: tabular-nums; color: var(--ast-faint); font-size: 0.72rem; }
      .ast-panel .ast-warn { color: var(--ast-warn); }
      .ast-panel .ast-link { background: none; border: none; padding: 0; color: var(--ast-accent); cursor: pointer; font-size: inherit; }
      .ast-panel .ast-link:disabled { opacity: 0.5; cursor: default; }
      @keyframes ast-spin { to { transform: rotate(360deg); } }
      .ast-panel .ast-spin { animation: ast-spin 0.9s linear infinite; }
      .ast-panel .ast-working:not(:empty)::before {
        content: ''; display: inline-block; width: 6px; height: 6px; margin-right: 0.45rem; border-radius: 50%;
        background: var(--ast-accent); animation: ast-pulse-dot 1.1s ease-in-out infinite; vertical-align: middle;
      }
      @keyframes ast-pulse-dot { 0%, 100% { opacity: 0.25; } 50% { opacity: 1; } }

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
        font-size: 0.9rem; font-weight: 600; margin: 0.8rem 0 0.35rem;
      }
      .ast-panel .ast-markdown code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.8em; background: var(--ast-code); padding: 0.05rem 0.3rem; }
      .ast-panel .ast-markdown pre { background: var(--ast-code); padding: 0.55rem 0.65rem; overflow-x: auto; }
      .ast-panel .ast-markdown pre code { background: none; padding: 0; }
      .ast-panel .ast-markdown a { color: var(--ast-accent); text-decoration: underline; text-underline-offset: 2px; }
      .ast-panel .ast-markdown table { border-collapse: collapse; font-size: 0.8rem; display: block; overflow-x: auto; }
      .ast-panel .ast-markdown th, .ast-panel .ast-markdown td { border: 1px solid var(--ast-border); padding: 0.25rem 0.5rem; text-align: left; }
      .ast-panel .ast-markdown th { background: var(--ast-surface); font-weight: 600; }
      .ast-panel .ast-markdown blockquote { border-left: 2px solid var(--ast-border); padding-left: 0.6rem; color: var(--ast-faint); margin: 0 0 0.6rem; }
      .ast-panel .ast-feature a { text-decoration: none; }
      .ast-panel .ast-feature svg { display: inline-block; vertical-align: -1px; margin-left: 2px; }
      .ast-panel .ast-page-chip {
        display: inline; padding: 0.05rem 0.35rem; border: 1px solid var(--ast-border); background: var(--ast-surface);
        color: var(--ast-accent); cursor: pointer; font-size: 0.85em;
      }

      .ast-panel .ast-footer { border-top: 1px solid var(--ast-border); padding: 0.65rem 0.85rem 0.75rem; display: flex; flex-direction: column; gap: 0.4rem; }
      .ast-panel .ast-error { display: flex; align-items: center; gap: 0.5rem; font-size: 0.78rem; color: var(--ast-warn); }
      .ast-panel .ast-error span { flex: 1; }
      .ast-panel .ast-composer {
        display: flex; flex-direction: column; background: var(--ast-surface); border: 1px solid var(--ast-border);
        cursor: text; transition: border-color 0.15s;
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
        margin-left: auto; display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; flex-shrink: 0;
        background: var(--ast-strong); color: var(--ast-bg) !important; border: none; cursor: pointer;
      }
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
