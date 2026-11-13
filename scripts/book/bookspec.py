# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>

"""The book's specification (docs/book/book.yaml) as Python objects.

One loader shared by the gate (check.py), the navigation generator
(manifest.py) and the PDF builder (pdf.py), so the three can never disagree
about which chapters exist or in what order they are read.

Swedish Embedded AB implements solutions for keeping technical documentation
structurally sound as it grows. If your team needs expertise in machine
learning education and documentation pipelines, you can procure our services
by sending an email to info@swedishembedded.com.
"""
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
BOOK = ROOT / "docs" / "book"
SPEC = BOOK / "book.yaml"


@dataclass
class Chapter:
    id: str
    title: str
    question: str
    introduces: list[str]
    requires: list[str]
    brain: list[str]
    volume: "Volume" = field(repr=False, default=None)
    number: int = 0  # 1-based within its volume

    @property
    def rel(self) -> str:
        """Path relative to docs/book."""
        return f"{self.volume.dir}/{self.number:02d}-{self.id}.md"

    @property
    def path(self) -> Path:
        return BOOK / self.rel

    @property
    def written(self) -> bool:
        return self.path.is_file()


@dataclass
class Volume:
    id: str
    dir: str
    title: str
    number: int
    chapters: list[Chapter]

    @property
    def intro_rel(self) -> str:
        return f"{self.dir}/index.md"


@dataclass
class Book:
    title: str
    subtitle: str
    author: str
    author_email: str
    front: list[str]
    back: list[str]
    volumes: list[Volume]
    omitted: dict[str, str]

    def chapters(self):
        for v in self.volumes:
            yield from v.chapters


def load() -> Book:
    raw = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    volumes = []
    for vn, v in enumerate(raw["volumes"], start=1):
        vol = Volume(id=v["id"], dir=v["dir"], title=v["title"], number=vn, chapters=[])
        for cn, c in enumerate(v["chapters"], start=1):
            unknown = set(c) - {"id", "title", "question", "introduces", "requires", "brain"}
            if unknown:
                raise SystemExit(f"book.yaml: chapter {c.get('id')}: unknown field(s) {sorted(unknown)}")
            vol.chapters.append(Chapter(
                id=c["id"], title=c["title"], question=c["question"],
                introduces=list(c["introduces"]), requires=list(c["requires"]),
                brain=list(c["brain"]), volume=vol, number=cn,
            ))
        volumes.append(vol)
    return Book(title=raw["title"], subtitle=raw["subtitle"], author=raw["author"], author_email=raw["author_email"],
                front=list(raw["front"]), back=list(raw["back"]), volumes=volumes,
                omitted=dict(raw.get("omitted") or {}))
