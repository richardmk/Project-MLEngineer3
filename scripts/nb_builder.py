"""Shared helpers to build notebooks from cell lists and fill interpretation cells.

A markdown cell whose source is an UPPERCASE key ending in INTERPRETATION is a
placeholder: it is emitted only once its text exists in the interpretations dict,
so notebooks can be executed first and narrated after reviewing the results.
"""

from pathlib import Path

import nbformat as nbf

Cells = list[tuple[str, str]]


def _is_placeholder(src: str) -> bool:
    return src.isupper() and src.endswith("INTERPRETATION")


def build(cells: Cells, interpretations: dict[str, str], out: Path) -> None:
    """Write a fresh (unexecuted) notebook; unfilled placeholders are skipped."""
    nb = nbf.v4.new_notebook()
    for kind, src in cells:
        if kind == "code":
            nb.cells.append(nbf.v4.new_code_cell(src))
        elif _is_placeholder(src):
            if src in interpretations:
                cell = nbf.v4.new_markdown_cell(interpretations[src])
                cell.metadata["interpretation"] = src
                nb.cells.append(cell)
        else:
            nb.cells.append(nbf.v4.new_markdown_cell(src))
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nbf.write(nb, out)
    print(f"Wrote {out}")


def fill(cells: Cells, interpretations: dict[str, str], path: Path) -> None:
    """Insert/update interpretation cells in an executed notebook, keeping its outputs.

    Each interpretation goes right after the code cell preceding its placeholder in `cells`.
    """
    nb = nbf.read(path, as_version=4)
    for pos, (kind, src) in enumerate(cells):
        if kind != "md" or not _is_placeholder(src) or src not in interpretations:
            continue
        anchor = next(s for k, s in reversed(cells[:pos]) if k == "code")
        idx = next(
            i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source == anchor
        )
        cell = nbf.v4.new_markdown_cell(interpretations[src])
        cell.metadata["interpretation"] = src
        nxt = nb.cells[idx + 1] if idx + 1 < len(nb.cells) else None
        if nxt is not None and nxt.metadata.get("interpretation") == src:
            nb.cells[idx + 1] = cell
        else:
            nb.cells.insert(idx + 1, cell)
    nbf.write(nb, path)
    print(f"Filled interpretations in {path}")
