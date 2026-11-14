/**
 * Page layout of the labs (Probe Lab, Geometry Lab): the split panes, panels, view tabs and train
 * button. Scoped under .lab-page on the page's root element.
 */

export function LabPageStyles() {
  return <style>{STYLES}</style>
}

const STYLES = `
  .lab-page {
    height: 125vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .lab-page .pl-layout { display: flex; flex: 1; margin-top: 76px; overflow: hidden; }
  .lab-page .pl-left { width: 26%; min-width: 320px; max-width: 420px; display: flex; flex-direction: column; overflow: hidden; }
  .lab-page .pl-right { flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden; }
  .lab-page .pl-scroll { flex: 1; overflow-y: auto; }

  .lab-page .pl-panel { padding: 1rem 1.25rem; border-bottom: 1px solid var(--pl-divider); }
  .lab-page .pl-panel h2,
  .lab-page .pl-card-title h3 {
    margin: 0 0 0.625rem;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .lab-page .pl-card-title h3 { margin: 0; }

  .lab-page .pl-view-tabs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.375rem; }
  .lab-page .pl-view-tab {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.375rem;
    padding: 0.5rem 0.375rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.04);
    color: inherit;
    font-size: 0.7rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    min-width: 0;
  }
  .lab-page .pl-view-tab .pl-icon { width: 12px; height: 12px; }
  .lab-page .pl-view-tab:hover { opacity: 0.85; }
  .lab-page .pl-view-tab.active { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.2); }
  [data-bg="light"] .lab-page .pl-view-tab { background: rgba(0, 0, 0, 0.04); border-color: rgba(0, 0, 0, 0.1); }
  [data-bg="light"] .lab-page .pl-view-tab.active { background: rgba(0, 0, 0, 0.08); border-color: rgba(0, 0, 0, 0.2); }

  .lab-page .pl-results-grid,
  .lab-page .pl-tools {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .lab-page .pl-results-grid > .pl-panel:nth-child(odd),
  .lab-page .pl-tools > .pl-panel:nth-child(odd) { border-right: 1px solid var(--pl-divider); }

  .lab-page .pl-error {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0.75rem 1.25rem 0;
    padding: 0.625rem 0.875rem;
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
    color: #ef4444;
    font-size: 0.8rem;
  }
  .lab-page .pl-error button { margin-left: auto; background: none; border: none; color: inherit; cursor: pointer; font-size: 0.75rem; }

  .lab-page .pl-train-button {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border: 1px solid transparent;
    background: #ffffff;
    color: #0a0a0a;
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
  }
  .lab-page .pl-train-button:hover:not(:disabled) { opacity: 0.85; }
  .lab-page .pl-train-button:disabled { opacity: 0.4; cursor: not-allowed; }
  .lab-page .pl-train-button .pl-spinner { border-color: rgba(0, 0, 0, 0.15); border-top-color: #0a0a0a; }
  [data-bg="light"] .lab-page .pl-train-button { background: #1d1d1f; color: #ffffff; }
  [data-bg="light"] .lab-page .pl-train-button .pl-spinner { border-color: rgba(255, 255, 255, 0.3); border-top-color: #ffffff; }

  @media (max-width: 1100px) {
    .lab-page { height: auto; overflow: visible; }
    .lab-page .pl-layout { flex-direction: column; overflow: visible; }
    .lab-page .pl-left { width: 100%; max-width: none; border-right: none !important; }
    .lab-page .pl-scroll { overflow: visible; }
  }
  @media (max-width: 800px) {
    .lab-page .pl-results-grid,
    .lab-page .pl-tools { grid-template-columns: 1fr; }
    .lab-page .pl-results-grid > .pl-panel:nth-child(odd),
    .lab-page .pl-tools > .pl-panel:nth-child(odd) { border-right: none; }
  }
`
