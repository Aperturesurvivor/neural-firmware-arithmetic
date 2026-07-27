from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLICATION = ROOT / "publication"
DIST = PUBLICATION / "dist"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_file(path: Path) -> Path:
    root = ROOT.resolve()
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked publication input: {path}")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Publication input escapes the repository: {path}") from error
    if not resolved.is_file():
        raise FileNotFoundError(path)
    return resolved


def write_zip(destination: Path, members: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, archive_name in members:
            archive.write(contained_file(source), archive_name)


def main() -> None:
    DIST.mkdir(parents=True, exist_ok=True)

    full_source = PUBLICATION / "arxiv_source" / "main.tex"
    full_pdf = ROOT / "paper_phase7" / "deterministic-neurons-qwen.pdf"
    workshop = PUBLICATION / "mathai2026"
    workshop_pdf = workshop / "main.pdf"

    arxiv_zip = DIST / "deterministic-neurons-qwen-arxiv-source.zip"
    write_zip(arxiv_zip, [(full_source, "main.tex")])

    preprint_pdf = DIST / "deterministic-neurons-qwen-preprint.pdf"
    shutil.copy2(contained_file(full_pdf), preprint_pdf)

    workshop_zip = DIST / "deterministic-neurons-mathai2026-source.zip"
    write_zip(
        workshop_zip,
        [
            (workshop / "main.tex", "main.tex"),
            (workshop / "neurips_2026.sty", "neurips_2026.sty"),
        ],
    )

    generated = [arxiv_zip, preprint_pdf, workshop_zip]
    if workshop_pdf.exists():
        workshop_dist_pdf = DIST / "deterministic-neurons-mathai2026-anonymous.pdf"
        shutil.copy2(contained_file(workshop_pdf), workshop_dist_pdf)
        generated.append(workshop_dist_pdf)

    manifest = {
        "generated_by": "publication/build_packages.py",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in generated
        ],
    }
    (DIST / "SHA256SUMS.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
