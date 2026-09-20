"""The report's stylesheet, kept apart from the layout that uses it.

One block of CSS for one file: the light values in `:root`, the dark ones stepped for the
dark surface under `prefers-color-scheme`. Nothing here is generated and nothing is
conditional, so the whole sheet can be read next to the markup it styles.

The eight series slots are a validated categorical palette in fixed order — an endpoint
keeps its slot across both probe charts, and past eight the report charts the first eight and
says so rather than inventing a ninth hue. The dark column is the same eight hues stepped
for the dark surface, not a second palette.
"""

CSS = """\
\
:root {
  color-scheme: light;
  --plane: #f9f9f7;
  --surface: #fcfcfb;
  --ink: #0b0b0b;
  --ink-2: #52514e;
  --muted: #74716a;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --rule: rgba(11, 11, 11, 0.10);
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
  --series-4: #eda100;
  --series-5: #e87ba4;
  --series-6: #008300;
  --series-7: #4a3aa7;
  --series-8: #e34948;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --plane: #0d0d0d;
    --surface: #1a1a19;
    --ink: #ffffff;
    --ink-2: #c3c2b7;
    --muted: #898781;
    --grid: #2c2c2a;
    --axis: #383835;
    --rule: rgba(255, 255, 255, 0.10);
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
    --series-4: #c98500;
    --series-5: #d55181;
    --series-6: #008300;
    --series-7: #9085e9;
    --series-8: #e66767;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--plane);
  color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.page { max-width: 1200px; margin: 0 auto; padding: 32px 24px 64px; overflow-wrap: anywhere; }
.head h1 { margin: 0 0 6px; font-size: 22px; }
.head .sub { margin: 0; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.head .where { margin: 4px 0 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
.head .method { margin: 8px 0 0; color: var(--ink-2); font-size: 13px; }
.answer { margin: 20px 0 4px; font-size: 16px; }
section { margin-top: 28px; }
h2 { margin: 0 0 10px; font-size: 15px; }
h3 { margin: 0 0 8px; font-size: 13px; font-weight: 600; color: var(--ink-2); }
section.table { margin-top: 20px; }
section.caveats { margin-top: 20px; }
.caption { margin: 0 0 8px; color: var(--ink-2); }
/* A figure follows the endpoint table directly, with no section of its own to space it. */
figure { margin: 24px 0; }
figcaption { margin: 8px 0 0; color: var(--muted); font-size: 12px; }
.scroll {
  overflow-x: auto;
  border: 1px solid var(--rule);
  border-radius: 6px;
  /* A right-edge fade that only shows while there is more to scroll to: the cover layer
     (`local`, so it scrolls with the table) sits at the far right of the full scrollable
     width and hides the shadow layer (`scroll`, pinned to the visible edge) until the
     table is scrolled away from its own end -- no script needed to know which is true. */
  background:
    linear-gradient(to left, var(--surface) 40%, rgba(0, 0, 0, 0)) 100% 0 / 24px 100% local
      no-repeat,
    linear-gradient(to left, var(--rule), rgba(0, 0, 0, 0)) 100% 0 / 12px 100% scroll no-repeat,
    var(--surface);
}
/* No horizontal scroll here: a chart scales to the page, only the tables scroll. */
.chart-scroll {
  padding: 12px;
  border: 1px solid var(--rule);
  border-radius: 6px;
  background: var(--surface);
}
/* 13 columns at the body's own size and padding overflowed a 1280px window by 100px, so the
   endpoint table's widest cell was cut off on the screen most readers open it in. A notch
   down on both buys back more than that and still reads as the same table. */
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 13px;
  font-variant-numeric: tabular-nums;
}
th, td { padding: 6px 7px; text-align: left; white-space: nowrap; }
thead th { border-bottom: 1px solid var(--rule); color: var(--ink-2); font-weight: 600; }
tbody tr + tr td { border-top: 1px solid var(--rule); }
/* The group header row above the endpoint table (cache / price / speed): a thin rule
   between groups, not around every cell, so the grouping reads as three bands. */
thead tr.groups th {
  border-bottom: none;
  padding-bottom: 2px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.02em;
  color: var(--muted);
}
thead tr.groups th.group-start { border-left: 1px solid var(--rule); }
details.turns { margin-top: 8px; }
details.turns summary {
  cursor: pointer;
  color: var(--ink-2);
  font-size: 13px;
  font-weight: 600;
}
details.turns section.table { margin-top: 12px; }
ul.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 16px;
  margin: 0 0 10px;
  padding: 0;
  list-style: none;
  color: var(--ink-2);
  font-size: 12px;
}
ul.legend li { display: flex; align-items: center; gap: 6px; }
.swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; }
.swatch.shade-light { background: var(--ink-2); opacity: 0.35; }
.swatch.shade-dark { background: var(--ink-2); }
ul.notes { margin: 0; padding-left: 20px; color: var(--ink-2); font-size: 13px; }
ul.notes li + li { margin-top: 4px; }
sup a { color: var(--ink-2); text-decoration: none; }
sup a:hover { text-decoration: underline; }
code { font-family: ui-monospace, monospace; }
svg.chart {
  display: block;
  width: 100%;
  height: auto;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
}
.s1 { fill: var(--series-1); background-color: var(--series-1); }
.s2 { fill: var(--series-2); background-color: var(--series-2); }
.s3 { fill: var(--series-3); background-color: var(--series-3); }
.s4 { fill: var(--series-4); background-color: var(--series-4); }
.s5 { fill: var(--series-5); background-color: var(--series-5); }
.s6 { fill: var(--series-6); background-color: var(--series-6); }
.s7 { fill: var(--series-7); background-color: var(--series-7); }
.s8 { fill: var(--series-8); background-color: var(--series-8); }
.bar.light { opacity: 0.35; }
.grid { stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--axis); stroke-width: 1; }
.line { fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; }
.marker { fill: var(--series-1); stroke: var(--surface); stroke-width: 2; }
.tick { fill: var(--muted); font-size: 12px; }
.tick.end { text-anchor: end; }
.tick.mid { text-anchor: middle; }
.row-label { fill: var(--ink-2); font-size: 12px; }
.value { fill: var(--ink-2); font-size: 12px; font-variant-numeric: tabular-nums; }
.axis-title { fill: var(--muted); font-size: 11px; text-anchor: end; }
@media (max-width: 900px) {
  .page { padding: 16px 12px 40px; }
  /* Scaled to a narrow column the ticks render unreadably small; scroll the chart instead. */
  .chart-scroll { overflow-x: auto; }
  .chart-scroll svg.chart { min-width: 640px; }
}
"""
