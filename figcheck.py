"""Automatic check that no two visible text elements of a figure overlap (titles, axis labels, tick labels,
annotations, legend entries, figure texts). Used by figures.py and thailand_map.py before saving."""


def text_overlaps(fig, tol=0.5):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    items = []

    def add(t):
        if t is None or not t.get_visible() or not str(t.get_text()).strip():
            return
        bb = t.get_window_extent(rend)
        if bb.width > 0 and bb.height > 0:
            items.append((str(t.get_text()).strip().replace("\n", " ")[:30], bb))

    def ticks(axis, lim):
        lo, hi = sorted(lim)
        return [lab for loc, lab in zip(axis.get_ticklocs(), axis.get_ticklabels()) if lo - 1e-9 <= loc <= hi + 1e-9]

    for ax in fig.axes:
        if not ax.get_visible():
            continue
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label, *ax.texts,
                  *ticks(ax.xaxis, ax.get_xlim()), *ticks(ax.yaxis, ax.get_ylim())):
            add(t)
        if ax.get_legend() is not None:
            for t in ax.get_legend().get_texts():
                add(t)
    for leg in fig.legends:
        for t in leg.get_texts():
            add(t)
    for t in fig.texts:
        add(t)
    hits = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            if min(a.x1, b.x1) - max(a.x0, b.x0) > tol and min(a.y1, b.y1) - max(a.y0, b.y0) > tol:
                hits.append(f"'{items[i][0]}' x '{items[j][0]}'")
    return hits
