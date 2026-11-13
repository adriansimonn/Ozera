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
      .ast-nav-button svg { width: 15px; height: 15px; }
      .ast-nav-button:hover { background: rgba(255, 255, 255, 0.06); }
      .ast-nav-button.active { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.22); }
      [data-bg="light"] .ast-nav-button { color: #1d1d1f; border-color: rgba(0, 0, 0, 0.12); }
      [data-bg="light"] .ast-nav-button:hover { background: rgba(0, 0, 0, 0.04); }
      [data-bg="light"] .ast-nav-button.active { background: rgba(0, 0, 0, 0.07); border-color: rgba(0, 0, 0, 0.2); }

      .ast-panel {
        --ast-bg: #0d0d0e; --ast-surface: rgba(255, 255, 255, 0.04); --ast-border: rgba(255, 255, 255, 0.1);
        --ast-text: #ffffff; --ast-faint: rgba(255, 255, 255, 0.55); --ast-user: rgba(255, 255, 255, 0.07);
        --ast-accent: rgba(96, 165, 250, 0.95); --ast-warn: #fab219; --ast-code: rgba(255, 255, 255, 0.08);
        position: fixed; top: 0; right: 0; height: 125vh; z-index: 120;
        display: flex; flex-direction: column; background: var(--ast-bg); color: var(--ast-text);
        border-left: 1px solid var(--ast-border); font-size: 0.875rem; line-height: 1.5;
      }
      [data-bg="light"] .ast-panel {
        --ast-bg: #ffffff; --ast-surface: rgba(0, 0, 0, 0.03); --ast-border: rgba(0, 0, 0, 0.1);
        --ast-text: #1d1d1f; --ast-faint: rgba(0, 0, 0, 0.55); --ast-user: rgba(0, 0, 0, 0.05);
        --ast-accent: #2563eb; --ast-warn: #b77900; --ast-code: rgba(0, 0, 0, 0.06);
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
      .ast-panel .ast-icon {
        display: inline-flex; align-items: center; justify-content: center; width: 28px; height: 28px; flex-shrink: 0;
        background: transparent; border: 1px solid transparent; cursor: pointer; color: var(--ast-faint);
      }
      .ast-panel .ast-icon svg { width: 15px; height: 15px; }
      .ast-panel .ast-icon:hover:not(:disabled) { color: var(--ast-text); background: var(--ast-surface); }
      .ast-panel .ast-icon:disabled { opacity: 0.4; cursor: default; }

      .ast-panel .ast-header { padding: 0.75rem 0.85rem 0.6rem; border-bottom: 1px solid var(--ast-border); display: flex; flex-direction: column; gap: 0.5rem; }
      .ast-panel .ast-header-row { display: flex; align-items: center; gap: 0.35rem; position: relative; }
      .ast-panel .ast-chat-title {
        flex: 1; min-width: 0; display: flex; align-items: center; gap: 0.35rem; padding: 0.3rem 0.4rem;
        background: transparent; border: 1px solid transparent; cursor: pointer; font-weight: 600; text-align: left;
      }
      .ast-panel .ast-chat-title span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-chat-title svg { width: 14px; height: 14px; flex-shrink: 0; color: var(--ast-faint); }
      .ast-panel .ast-chat-title:hover:not(:disabled) { background: var(--ast-surface); }
      .ast-panel .ast-model { justify-content: space-between; }
      .ast-panel .ast-model > div:first-child { flex: 1; min-width: 0; }
      .ast-panel .ast-chat-cost { color: var(--ast-faint); font-size: 0.75rem; white-space: nowrap; font-variant-numeric: tabular-nums; }

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
      .ast-panel .ast-menu-item svg { width: 14px; height: 14px; }
      .ast-panel .ast-menu-item:hover { background: var(--ast-surface); }
      .ast-panel .ast-menu-title { min-width: 0; justify-content: space-between; }
      .ast-panel .ast-menu-title span:first-child { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-menu-head { padding: 0.35rem 0.5rem 0.25rem; font-size: 0.72rem; color: var(--ast-faint); }
      .ast-panel .ast-menu-foot { padding: 0.35rem 0.5rem 0.25rem; font-size: 0.72rem; color: var(--ast-faint); border-top: 1px solid var(--ast-border); margin-top: 0.2rem; font-variant-numeric: tabular-nums; }
      .ast-panel .ast-menu-check { display: inline-block; width: 14px; height: 14px; flex-shrink: 0; }
      .ast-panel .ast-allowance { position: relative; flex-shrink: 0; }
      .ast-panel .ast-allowance-button {
        display: inline-flex; align-items: center; gap: 0.3rem; height: 30px; padding: 0 0.5rem; cursor: pointer;
        background: transparent; border: 1px solid var(--ast-border); color: var(--ast-faint); font-size: 0.75rem;
        font-variant-numeric: tabular-nums; white-space: nowrap;
      }
      .ast-panel .ast-allowance-button svg { width: 13px; height: 13px; }
      .ast-panel .ast-allowance-button:hover { color: var(--ast-text); background: var(--ast-surface); }
      .ast-panel .ast-allowance-button.on { color: var(--ast-accent); border-color: var(--ast-accent); }
      .ast-panel .ast-allowance-menu { left: auto; right: 0; width: 200px; }
      .ast-panel .ast-rename {
        flex: 1; min-width: 0; padding: 0.3rem 0.45rem; background: var(--ast-surface); color: var(--ast-text);
        border: 1px solid var(--ast-border); font: inherit; outline: none;
      }

      .ast-panel .ast-body { flex: 1; overflow-y: auto; padding: 0.85rem; display: flex; flex-direction: column; gap: 1.1rem; }
      .ast-panel .ast-empty { margin: auto 0; display: flex; flex-direction: column; align-items: stretch; gap: 0.45rem; }
      .ast-panel .ast-empty > svg { width: 22px; height: 22px; color: var(--ast-faint); align-self: center; margin-bottom: 0.4rem; }
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
      .ast-panel .ast-tools { display: flex; flex-wrap: wrap; gap: 0.3rem; }
      .ast-panel .ast-tool {
        display: inline-flex; align-items: center; gap: 0.3rem; max-width: 100%; padding: 0.15rem 0.45rem;
        font-size: 0.72rem; color: var(--ast-faint); border: 1px solid var(--ast-border); background: var(--ast-surface);
      }
      .ast-panel .ast-tool span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .ast-panel .ast-tool svg { width: 11px; height: 11px; flex-shrink: 0; }
      .ast-panel .ast-tool--error { color: var(--ast-warn); }
      .ast-panel .ast-tool--run { color: var(--ast-text); }
      .ast-panel .ast-tool--skipped { opacity: 0.6; }
      .ast-panel .ast-tool .ast-tool-cost { color: var(--ast-faint); font-variant-numeric: tabular-nums; flex-shrink: 0; }
      .ast-panel button.ast-tool--open { cursor: pointer; font-size: 0.72rem; }
      .ast-panel button.ast-tool--open:hover { border-color: var(--ast-accent); }

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
      .ast-panel .ast-approval--auto { border-color: var(--ast-border); }
      .ast-panel .ast-approval--auto .ast-approval-run { border-bottom: none; }
      .ast-panel .ast-approval--auto .ast-approval-price svg { width: 13px; height: 13px; vertical-align: -2px; }
      .ast-panel .ast-approval-auto-foot { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
      .ast-panel .ast-approval-auto-foot .ast-approval-note { margin-right: auto; }
      .ast-panel .ast-approval-footer { display: flex; align-items: center; gap: 0.6rem; padding: 0.55rem 0.7rem; font-size: 0.78rem; flex-wrap: wrap; }
      .ast-panel .ast-approval-footer > span { font-variant-numeric: tabular-nums; }
      .ast-panel .ast-approval-actions { margin-left: auto; display: flex; gap: 0.4rem; }
      .ast-panel .ast-approval-skip, .ast-panel .ast-approval-go {
        display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.35rem 0.8rem; cursor: pointer; font-size: 0.8rem; font-weight: 500;
        border: 1px solid var(--ast-border); background: transparent;
      }
      .ast-panel .ast-approval-skip:hover:not(:disabled) { background: var(--ast-surface); }
      .ast-panel .ast-approval-go { background: var(--ast-text); color: var(--ast-bg) !important; border-color: transparent; }
      .ast-panel .ast-approval-go svg { width: 12px; height: 12px; }
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
      .ast-panel .ast-composer { display: flex; align-items: flex-end; gap: 0.45rem; }
      .ast-panel .ast-composer textarea {
        flex: 1; resize: none; min-height: 2.6rem; max-height: 12rem; padding: 0.5rem 0.6rem; font: inherit;
        background: var(--ast-surface); color: var(--ast-text); border: 1px solid var(--ast-border); outline: none;
        field-sizing: content;
      }
      .ast-panel .ast-composer textarea:focus { border-color: var(--ast-accent); }
      .ast-panel .ast-send {
        display: inline-flex; align-items: center; justify-content: center; width: 34px; height: 34px; flex-shrink: 0;
        background: var(--ast-text); color: var(--ast-bg) !important; border: none; cursor: pointer;
      }
      .ast-panel .ast-send svg { width: 16px; height: 16px; }
      .ast-panel .ast-send:disabled { opacity: 0.35; cursor: default; }
      .ast-panel .ast-footnote { font-size: 0.72rem; color: var(--ast-faint); }
      .ast-panel .ast-footnote .ast-chat-cost { white-space: normal; font-size: inherit; }
    `}</style>
  )
}
