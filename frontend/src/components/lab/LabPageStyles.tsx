/**
 * Page layout of the labs (Probe Lab, Geometry Lab), over the app's page frame (styles/app.css:
 * .oz-page, .oz-side, .oz-section): the results column's panels and grids. Scoped under .lab-page
 * on the page's root element.
 */

export function LabPageStyles() {
  return <style>{STYLES}</style>
}

const STYLES = `
  .lab-page .pl-panel { padding: 1.35rem var(--oz-gutter) 1.5rem; border-bottom: 1px solid var(--oz-rule); }
  .lab-page .pl-view-tabs { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }

  .lab-page .pl-results-grid,
  .lab-page .pl-tools {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .lab-page .pl-results-grid > .pl-panel:nth-child(odd),
  .lab-page .pl-tools > .pl-panel:nth-child(odd) { border-right: 1px solid var(--oz-rule); }

  .lab-page .pl-error { margin: 1rem var(--oz-gutter) 0; }

  /* A view with nothing to show yet: its pictogram in the middle of the column */
  .lab-page .oz-main .pl-panel:only-child:has(> .oz-empty:only-child) {
    display: grid;
    place-items: center;
    min-height: calc(125vh - var(--oz-nav-h) - 1px);
    border-bottom: none;
  }

  @media (max-width: 1100px) {
    .lab-page.oz-page { height: auto; overflow: visible; }
    .lab-page .oz-page__body { flex-direction: column; overflow: visible; }
    .lab-page .oz-side { width: 100%; max-width: none; border-right: none; }
    .lab-page .oz-scroll { overflow: visible; }
  }
  @media (max-width: 800px) {
    .lab-page .pl-results-grid,
    .lab-page .pl-tools { grid-template-columns: 1fr; }
    .lab-page .pl-results-grid > .pl-panel:nth-child(odd),
    .lab-page .pl-tools > .pl-panel:nth-child(odd) { border-right: none; }
  }
`
