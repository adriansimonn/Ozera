/**
 * Page layout of the labs (Probe Lab, Geometry Lab) in each interface: the glass page's header,
 * view tabs, setup column and panels, and the default page's split panes. Scoped under
 * .lab-page--glass / .lab-page--default on the page's root element.
 */

export function LabPageStyles({ variant }: { variant: 'glass' | 'default' }) {
  return <style>{variant === 'glass' ? GLASS : DEFAULT}</style>
}

const GLASS = `
  .lab-page--glass {
    min-height: 125vh;
    padding-top: 70px;
    color: #ffffff;
  }
  [data-bg="light"] .lab-page--glass { color: #1d1d1f; }

  .lab-page--glass .pl-content { max-width: 1800px; margin: 0 auto; padding: 2rem; }
  .lab-page--glass .pl-header { margin-bottom: 1.5rem; }
  .lab-page--glass .pl-header h1 { margin: 0; font-size: 1.75rem; font-weight: 600; letter-spacing: -0.02em; }
  .lab-page--glass .pl-header p { margin: 0.5rem 0 0; font-size: 0.9rem; }

  .lab-page--glass .pl-view-tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; }
  .lab-page--glass .pl-view-tab {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.625rem 1rem;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #ffffff;
    font-size: 0.875rem;
    cursor: pointer;
    transition: all 0.2s;
  }
  .lab-page--glass .pl-view-tab .pl-icon { width: 16px; height: 16px; }
  .lab-page--glass .pl-view-tab:hover { background: rgba(255, 255, 255, 0.05); border-color: rgba(255, 255, 255, 0.2); }
  .lab-page--glass .pl-view-tab.active { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.4); }
  [data-bg="light"] .lab-page--glass .pl-view-tab { background: rgba(0, 0, 0, 0.04); border-color: rgba(0, 0, 0, 0.1); color: #1d1d1f; }
  [data-bg="light"] .lab-page--glass .pl-view-tab:hover { background: rgba(0, 0, 0, 0.06); border-color: rgba(0, 0, 0, 0.15); }
  [data-bg="light"] .lab-page--glass .pl-view-tab.active { background: rgba(0, 0, 0, 0.08); border-color: rgba(0, 0, 0, 0.25); }

  .lab-page--glass .pl-error {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.875rem 1rem;
    margin-bottom: 1.5rem;
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: rgba(239, 68, 68, 0.95);
    font-size: 0.85rem;
  }
  .lab-page--glass .pl-error button {
    margin-left: auto;
    padding: 0.25rem 0.75rem;
    background: transparent;
    border: 1px solid rgba(239, 68, 68, 0.3);
    color: inherit;
    cursor: pointer;
  }

  .lab-page--glass .pl-layout { display: grid; grid-template-columns: 370px minmax(0, 1fr); gap: 1.5rem; align-items: start; }

  .lab-page--glass .pl-panel {
    margin-bottom: 1.5rem;
    padding: 1.25rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.1);
  }
  [data-bg="light"] .lab-page--glass .pl-panel { background: rgba(0, 0, 0, 0.03); border-color: rgba(0, 0, 0, 0.1); }
  [data-bg="light"][data-interface="glass"] .lab-page--glass .pl-panel { background: rgba(255, 255, 255, 0.55); }
  .lab-page--glass .pl-panel h2 {
    margin: 0 0 1rem;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .lab-page--glass .pl-results-grid,
  .lab-page--glass .pl-tools {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1.5rem;
  }
  .lab-page--glass .pl-results-grid .pl-panel,
  .lab-page--glass .pl-tools .pl-panel { margin-bottom: 0; }
  .lab-page--glass .pl-results-grid { margin-bottom: 1.5rem; }

  .lab-page--glass .pl-train-button {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    padding: 0.9rem 1.25rem;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.25);
    color: #ffffff;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
  }
  .lab-page--glass .pl-train-button:hover:not(:disabled) { background: rgba(255, 255, 255, 0.15); border-color: rgba(255, 255, 255, 0.4); }
  .lab-page--glass .pl-train-button:disabled { opacity: 0.5; cursor: not-allowed; }
  .lab-page--glass .pl-train-button .pl-spinner { border-color: rgba(255, 255, 255, 0.3); border-top-color: #ffffff; }
  [data-bg="light"] .lab-page--glass .pl-train-button { background: rgba(0, 0, 0, 0.06); border-color: rgba(0, 0, 0, 0.15); color: #1d1d1f; }
  [data-bg="light"] .lab-page--glass .pl-train-button:hover:not(:disabled) { background: rgba(0, 0, 0, 0.1); }
  [data-bg="light"] .lab-page--glass .pl-train-button .pl-spinner { border-color: rgba(0, 0, 0, 0.15); border-top-color: #1d1d1f; }

  @media (max-width: 1200px) {
    .lab-page--glass .pl-layout { grid-template-columns: 1fr; }
  }
  @media (max-width: 900px) {
    .lab-page--glass .pl-content { padding: 1rem; }
    .lab-page--glass .pl-view-tabs { flex-wrap: wrap; }
    .lab-page--glass .pl-view-tab { flex: 1; min-width: 120px; justify-content: center; }
    .lab-page--glass .pl-results-grid,
    .lab-page--glass .pl-tools { grid-template-columns: 1fr; }
  }
`

const DEFAULT = `
  .lab-page--default {
    height: 125vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .lab-page--default .pl-layout { display: flex; flex: 1; margin-top: 76px; overflow: hidden; }
  .lab-page--default .pl-left { width: 26%; min-width: 320px; max-width: 420px; display: flex; flex-direction: column; overflow: hidden; }
  .lab-page--default .pl-right { flex: 1; min-width: 0; display: flex; flex-direction: column; overflow: hidden; }
  .lab-page--default .pl-scroll { flex: 1; overflow-y: auto; }

  .lab-page--default .pl-panel { padding: 1rem 1.25rem; border-bottom: 1px solid var(--pl-divider); }
  .lab-page--default .pl-panel h2,
  .lab-page--default .pl-card-title h3 {
    margin: 0 0 0.625rem;
    font-size: 0.65rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .lab-page--default .pl-card-title h3 { margin: 0; }

  .lab-page--default .pl-view-tabs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.375rem; }
  .lab-page--default .pl-view-tab {
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
  .lab-page--default .pl-view-tab .pl-icon { width: 12px; height: 12px; }
  .lab-page--default .pl-view-tab:hover { opacity: 0.85; }
  .lab-page--default .pl-view-tab.active { background: rgba(255, 255, 255, 0.1); border-color: rgba(255, 255, 255, 0.2); }
  [data-bg="light"] .lab-page--default .pl-view-tab { background: rgba(0, 0, 0, 0.04); border-color: rgba(0, 0, 0, 0.1); }
  [data-bg="light"] .lab-page--default .pl-view-tab.active { background: rgba(0, 0, 0, 0.08); border-color: rgba(0, 0, 0, 0.2); }

  .lab-page--default .pl-results-grid,
  .lab-page--default .pl-tools {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .lab-page--default .pl-results-grid > .pl-panel:nth-child(odd),
  .lab-page--default .pl-tools > .pl-panel:nth-child(odd) { border-right: 1px solid var(--pl-divider); }

  .lab-page--default .pl-error {
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
  .lab-page--default .pl-error button { margin-left: auto; background: none; border: none; color: inherit; cursor: pointer; font-size: 0.75rem; }

  .lab-page--default .pl-train-button {
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
  .lab-page--default .pl-train-button:hover:not(:disabled) { opacity: 0.85; }
  .lab-page--default .pl-train-button:disabled { opacity: 0.4; cursor: not-allowed; }
  .lab-page--default .pl-train-button .pl-spinner { border-color: rgba(0, 0, 0, 0.15); border-top-color: #0a0a0a; }
  [data-bg="light"] .lab-page--default .pl-train-button { background: #1d1d1f; color: #ffffff; }
  [data-bg="light"] .lab-page--default .pl-train-button .pl-spinner { border-color: rgba(255, 255, 255, 0.3); border-top-color: #ffffff; }

  @media (max-width: 1100px) {
    .lab-page--default { height: auto; overflow: visible; }
    .lab-page--default .pl-layout { flex-direction: column; overflow: visible; }
    .lab-page--default .pl-left { width: 100%; max-width: none; border-right: none !important; }
    .lab-page--default .pl-scroll { overflow: visible; }
  }
  @media (max-width: 800px) {
    .lab-page--default .pl-results-grid,
    .lab-page--default .pl-tools { grid-template-columns: 1fr; }
    .lab-page--default .pl-results-grid > .pl-panel:nth-child(odd),
    .lab-page--default .pl-tools > .pl-panel:nth-child(odd) { border-right: none; }
  }
`
