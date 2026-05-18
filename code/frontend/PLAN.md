# Frontend PLAN.md — RaceScope React/Vite SPA

This is the local frontend copy of the master execution plan. For full context on agent roles, QA protocol, and phase ordering, read the root `PLAN.md` first.

**Frontend agents:** `FE-LAYOUT`, `FE-A11Y`, `FE-SYSTEM`
**QA agent** reviews this area after each phase.

---

## Quick Reference — Frontend TO_DO Items

All items are from `TO_DO.md`. Ownership assigned here.

| TO_DO Item | Agent | Phase |
|---|---|---|
| Fix vertical overflow on short-screen desktops | FE-LAYOUT | 3 |
| Add scroll/fade indicator to strategy-strip | FE-LAYOUT | 3 |
| Clamp hover tooltip at chart edges | FE-LAYOUT | 3 |
| Intermediate breakpoint (920–1200px) for rail | FE-LAYOUT | 3 |
| Mobile padding on context bubble & full-width CTA | FE-LAYOUT | 3 |
| Increase minimum font sizes (sub-12px elements) | FE-A11Y | 3 |
| Verify and fix muted-text contrast ratios | FE-A11Y | 3 |
| Disambiguate DriverRow headers (Piloto 1 / 2) | FE-A11Y | 3 |
| Fix incorrect ARIA roles in StrategyStrip | FE-A11Y | 3 |
| Add role="tab" and aria-selected to TopTabs | FE-A11Y | 3 |
| Add aria-live regions for async state changes | FE-A11Y | 3 |
| Add touch support for chart tooltips | FE-A11Y | 3 |
| Minimum touch target size for icon-btn | FE-A11Y | 3 |
| Unify compound color definitions (3 → 1 source) | FE-SYSTEM | 4 |
| Externalize TEAM_COLORS to shared constants | FE-SYSTEM | 4 |
| Remove orphaned/dead components | FE-SYSTEM | 4 |
| Lift App.jsx state into useReducer | FE-SYSTEM | 4 |
| Show delta-to-best in strategy cards | FE-SYSTEM | 4 |
| Add per-row retry button on error | FE-SYSTEM | 4 |

Not in Phase 3/4 scope (open design questions to decide first):
- Split `styles.css` into component files — deferred; decide after FE-SYSTEM finishes
- `force_recompute` toggle — deferred pending backend live-data completion
- Unimplemented tab visibility — deferred pending product decision

---

## Setup Checklist (all frontend agents run this first)

```bash
cd code/frontend
node --version   # must be ≥18
npm install
npm run dev      # starts Vite on :5173
# Open http://localhost:5173 in browser
# Verify Home tab loads and Pre-race tab shows driver selectors
```

If `npm install` fails:
```bash
rm -rf node_modules package-lock.json
npm install
```

---

## Shared File Coordination

Phases 3 agents (`FE-LAYOUT` and `FE-A11Y`) run in parallel and both touch `styles.css` and `StrategyCurveChart.jsx`.

**Rule:** each agent must read the current file state immediately before writing. Do not cache file contents between tasks. When both agents edit the same file, the second agent to write must include all changes from the first.

**Conflict resolution order by file:**

| File | FE-LAYOUT edits | FE-A11Y edits | Correct approach |
|---|---|---|---|
| `styles.css` | Layout rules, new breakpoints, padding | Font sizes, contrast, `.sr-only` | FE-A11Y reads FE-LAYOUT's version and adds on top |
| `StrategyCurveChart.jsx` | Tooltip x-clamp (line ~346) | `onMouseMove` → `onPointerMove` (line ~287) | Different lines — both can apply independently; second agent reads and preserves first's change |
| `StrategyStrip.jsx` | Wrapper div for fade | ARIA roles change | FE-A11Y reads FE-LAYOUT's version; wrap div must not break aria structure |

---

## FE-LAYOUT Detailed Instructions

See root `PLAN.md` Phase 3 → FE-LAYOUT section for full task list.

### Key constraints

- The `fixedDesktopRowHeight` variable in `App.jsx` and the `rowHeight` prop on `DriverRow` must be removed cleanly — check that `DriverRow` does not use `rowHeight` anywhere before removing the prop.
- The strategy-strip wrapper div change is minimal: only add `<div className="strategy-strip-wrapper">` around the existing div. Do not restructure the component beyond that.
- The tooltip clamp in `StrategyCurveChart.jsx` is a JS-side change to the `style` attribute on `.curve-tooltip`. It is 2 lines: compute `clampedLeft`, apply it. Do not change any other logic in the file.
- The `@media (max-width: 1080px)` breakpoint must be inserted between the existing `1200px` and `920px` blocks in `styles.css`, not at the end.

### How to test

```bash
# Run the dev server
npm run dev

# In browser DevTools:
# 1. Open responsive mode
# 2. Set viewport to 768px × 768px — verify both driver rows are visible without clipping
# 3. Set viewport to 375px × 812px — verify CTA is full-width and bubble has padding
# 4. Set viewport to 1024px × 768px — verify driver rail collapses to top strip (1080px breakpoint fires)
# 5. Run a strategy calculation (requires backend running on :8000)
#    Hover near lap 1 on any chart — tooltip should not clip left edge
#    Hover near final lap — tooltip should not clip right edge
```

---

## FE-A11Y Detailed Instructions

See root `PLAN.md` Phase 3 → FE-A11Y section for full task list.

### Key constraints

- The `.sr-only` class must be added to `styles.css`, not inline in a component.
- `aria-live="polite"` — use `polite`, not `assertive`. `assertive` interrupts screen reader flow mid-sentence and is intrusive for non-critical status updates.
- For `StrategyStrip`, the `<div role="listitem">` wrapper around each `<button>` will make the button a child of a listitem. Verify the button still receives click and keyboard events normally — it will, since `div[role=listitem]` does not intercept events.
- For `TopTabs`, `aria-selected` takes a boolean but HTML attributes are strings. In JSX, `aria-selected={true}` renders as `aria-selected="true"` which is correct.
- The `onPointerMove` change in `StrategyCurveChart.jsx` is a drop-in replacement: pointer events fire for mouse, touch, and stylus. The `onPointerLeave` replacement for `onMouseLeave` also works correctly across input types.
- `icon-btn` size increase to 36×36px: verify visually that it does not break the `row-head` flex layout in `DriverRow`. If it does, use `min-width`/`min-height` instead of fixed width/height.

### How to test

```bash
npm run dev

# Accessibility testing:
# 1. Open Chrome DevTools → Lighthouse → run Accessibility audit
#    Target: ≥ 85 score (Phase 3 target; ≥ 90 is Phase 5 target)
# 2. Open DevTools → Elements, inspect TopTabs buttons — verify aria-selected attribute present
# 3. Open DevTools → Elements, inspect StrategyStrip — verify role="list" and role="listitem"
# 4. Tab through the Pre-race page with keyboard only — verify all interactive elements are reachable
# 5. On a touch device or DevTools touch emulation:
#    Load Pre-race, run calculation, touch a chart — verify tooltip appears on touch

# Font size verification:
# DevTools → Elements → computed styles on .axis-caption, .data-mode-chip, .legend-item
# All must show font-size ≥ 11px

# Contrast verification (manual):
# Use WebAIM contrast checker with:
#   Dark theme muted: #b0bac6 on #16181b — should be ≥ 5.0:1
#   Light theme muted: #667587 on #ffffff — should be ≥ 5.3:1
```

---

## FE-SYSTEM Detailed Instructions

See root `PLAN.md` Phase 4 → FE-SYSTEM section for full task list.

### Key constraints

**On compound color unification:**
- Do NOT attempt to read CSS custom property values from JS at runtime via `getComputedStyle`. This adds complexity with no benefit since the CSS variables and the JS constants serve different consumers (CSS classes vs. chart rendering).
- The comment in `styles.css` is sufficient as the co-location note.

**On dead component deletion:**
- Run the grep verification command before deleting anything. If a component is imported anywhere (even in a test file or another dead component), do not delete it — remove the import chain first.
- After deletion, run `npm run build` to confirm no broken imports.

**On the `useReducer` refactor:**
- The `initialState` object in `appReducer.js` must match exactly the current `useState` default values in `App.jsx`. Do a side-by-side comparison.
- `useEffect` hooks in `App.jsx` that call `setXxx` functions must be updated to call `dispatch({ type: SET_XXX, payload: ... })`.
- Keep the `useMemo` and `useCallback` hooks unchanged — they are derived values, not state.
- After refactoring, the component must be functionally identical to before. Run the app and verify all interactions still work before marking complete.

**On the delta-to-best display:**
- `formatDelta` must handle edge cases: `seconds = 0` → `"BEST"`, `seconds < 0` (shouldn't happen but guard) → `"BEST"`, `NaN/Infinity` → `"—"`.
- In `StrategyStrip`, the current `<span className="time">` shows `formatRaceDuration`. Replace it with: delta for non-first strategies, absolute time for the first (best) strategy. Do not show both — space is limited.

**On the per-row retry:**
- The fetch logic extracted from `runPreRace` in `App.jsx` for single-row retry must use the same `force_recompute: true` flag as the full run.
- The `onRetry` prop must be added to `DriverRow`'s prop list. It is optional (`onRetry?`); if not provided, do not render the button.

### How to test

```bash
npm run build   # Must complete with 0 errors and 0 unresolved imports
npm run dev

# Verify compound color source:
# grep -r "COMPOUND_COLORS" src/  — should only appear in src/constants/compounds.js and one import in StrategyCurveChart.jsx
# grep -r "TEAM_COLORS" src/      — should only appear in src/constants/teams.js and one import in DriverRow.jsx

# Verify dead components gone:
# ls src/components/ — should NOT include ComparisonView, ControlPanel, DegradationChart, StrategyCard, StrategyTimeline

# Verify delta display:
# Run a strategy calculation. In the strategy strip, the first card should show the absolute time.
# The second, third etc. should show "+Xs" or "+Xm Xs" deltas.

# Verify reducer:
# Open React DevTools → Components → App
# State should appear as a single object (useReducer), not as multiple useState entries

# Verify per-row retry:
# Temporarily break the backend (stop uvicorn), press Calcular, verify error state appears.
# Restart backend, press per-row "Reintentar" — only that row's strategy re-runs.
```

---

## QA Checklist — Frontend

After Phase 3 (FE-LAYOUT + FE-A11Y):
- [ ] 768px-tall viewport: both driver rows visible without clipping
- [ ] 375px viewport: CTA button is full-width, bubble has adequate padding
- [ ] 1024px viewport: driver rail collapses above the strategy area (1080px breakpoint)
- [ ] Tooltip does not clip at left or right chart edges
- [ ] Strategy strip shows fade gradient when cards overflow
- [ ] All font sizes ≥ 11px (check computed styles in DevTools)
- [ ] TopTabs buttons have `aria-selected` attribute
- [ ] StrategyStrip has `role="list"` (not `listbox`)
- [ ] DriverRow shows "Piloto 1" / "Piloto 2" headings
- [ ] aria-live region present in DriverRow (check Elements panel)
- [ ] Chart tooltip appears on touch/pointer events (not just mouse)
- [ ] No console errors or warnings introduced
- [ ] Lighthouse accessibility score ≥ 85

After Phase 4 (FE-SYSTEM):
- [ ] `npm run build` completes with no errors
- [ ] No dead component files in `src/components/`
- [ ] `COMPOUND_COLORS` defined in exactly one place
- [ ] `TEAM_COLORS` defined in exactly one place
- [ ] Strategy strip shows "BEST" for first strategy, "+Xs" for others
- [ ] Error state shows per-row retry button
- [ ] App.jsx uses `useReducer` (verify in React DevTools)
- [ ] All Phase 3 improvements still present (no regression from FE-SYSTEM changes)
- [ ] Lighthouse accessibility score ≥ 88
