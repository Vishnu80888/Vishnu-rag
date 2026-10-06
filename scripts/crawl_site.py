"""Refresh data/docs from the live website (pure Python: httpx + stdlib).
Run on your machine:  python run.py crawl
Note: lists loaded by JavaScript (individual universities/courses/advisors/expos) are not in the static HTML.
"""
import re
from html.parser import HTMLParser
from pathlib import Path

import httpx

BASE = "https://www.codejobz.com"
PAGES = {"home": "/", "about": "/about", "universities": "/universities",
         "courses": "/courses", "advisors": "/advisors", "expos": "/expos"}
OUT = Path(__file__).resolve().parent.parent / "data" / "docs" / "crawled"


class Text(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "svg", "noscript", "header"}
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "section", "br", "span", "a"}

    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag in {"h1", "h2", "h3"}:
            self.out.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.out.append(data.strip() + " ")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=True, headers={"User-Agent": "Profi/1.0"}) as c:
        for name, path in PAGES.items():
            r = c.get(BASE + path)
            r.raise_for_status()
            p = Text()
            p.feed(r.text)
            body = re.sub(r"\n{3,}", "\n\n", "".join(p.out)).strip()
            (OUT / f"{name}.md").write_text(f"<!-- source: {BASE}{path} | page: {name} -->\n{body}\n", encoding="utf-8")
            print("saved", name, len(body), "chars")


if __name__ == "__main__":
    main()
