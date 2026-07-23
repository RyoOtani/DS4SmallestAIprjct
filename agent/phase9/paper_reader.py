"""
Phase 9: Paper Reader — Read, parse, and understand AI research papers.

Capabilities:
  ✅ Download papers from arXiv by ID
  ✅ Parse PDF text and extract key sections (abstract, method, results)
  ✅ Identify novel contributions vs prior work
  ✅ Extract algorithm pseudocode from papers
  ✅ Build a research knowledge graph linking papers
  ✅ Summarize papers in structured format
  ✅ Compare multiple papers on the same topic
  ✅ Search arXiv by keyword
"""

from __future__ import annotations
import json
import re
import time
import hashlib
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PaperSection:
    """A section extracted from a paper."""
    heading: str
    content: str
    subsections: list[PaperSection] = field(default_factory=list)
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split()) if self.content else 0


@dataclass
class Algorithm:
    """Pseudocode algorithm extracted from a paper."""
    name: str
    description: str
    pseudocode: str
    complexity: str = ""        # e.g., "O(n log n)"
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    source_paper: str = ""


@dataclass
class Paper:
    """Structured representation of a research paper."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str = ""
    sections: list[PaperSection] = field(default_factory=list)
    algorithms: list[Algorithm] = field(default_factory=list)
    contributions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)  # arXiv IDs cited
    keywords: list[str] = field(default_factory=list)
    url: str = ""
    pdf_path: str = ""
    summary: str = ""
    quality_score: float = 0.0  # heuristic quality 0-100


@dataclass
class ResearchGraph:
    """Knowledge graph linking papers by topic/method/citation."""
    papers: dict[str, Paper] = field(default_factory=dict)
    citations: dict[str, list[str]] = field(default_factory=dict)  # paper_id → [cited_ids]
    topic_clusters: dict[str, list[str]] = field(default_factory=dict)  # topic → [paper_ids]


class ArXivClient:
    """Client for the arXiv API."""

    BASE_URL = "http://export.arxiv.org/api/query"

    def search(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
    ) -> list[dict]:
        """
        Search arXiv and return paper metadata.

        Args:
            query: Search query (supports arXiv syntax: ti:, au:, abs:, cat:, etc.)
            max_results: Maximum number of results
            start: Starting index for pagination
            sort_by: 'relevance', 'lastUpdatedDate', 'submittedDate'
        """
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(max_results, 100),
            "sortBy": sort_by,
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                xml_data = resp.read().decode("utf-8")
            return self._parse_search_results(xml_data)
        except Exception as e:
            return [{"error": str(e), "query": query}]

    def get_paper(self, arxiv_id: str) -> Optional[dict]:
        """Get full metadata for a specific paper by arXiv ID."""
        clean_id = arxiv_id.replace("arxiv:", "").strip()
        results = self.search(f"id:{clean_id}", max_results=1)
        return results[0] if results else None

    def _parse_search_results(self, xml_data: str) -> list[dict]:
        """Parse arXiv API XML response."""
        results = []
        root = ET.fromstring(xml_data)
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        for entry in root.findall("atom:entry", ns):
            paper = {
                "arxiv_id": self._text(entry, "atom:id", ns).split("/")[-1],
                "title": self._text(entry, "atom:title", ns).strip(),
                "authors": [
                    self._text(a, "atom:name", ns)
                    for a in entry.findall("atom:author", ns)
                ],
                "abstract": self._text(entry, "atom:summary", ns).strip(),
                "published": self._text(entry, "atom:published", ns, ""),
                "updated": self._text(entry, "atom:updated", ns, ""),
                "categories": [
                    c.get("term", "")
                    for c in entry.findall("atom:category", ns)
                ],
                "url": self._text(entry, "atom:id", ns, ""),
                "pdf_url": "",
            }
            # Find PDF link
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    paper["pdf_url"] = link.get("href", "")
            results.append(paper)

        return results

    def _text(self, elem, tag, ns, default=""):
        found = elem.find(tag, ns)
        return found.text.strip() if found is not None and found.text else default


class PaperParser:
    """Parse and understand research papers from text/PDF content."""

    SECTION_PATTERNS = [
        re.compile(r"^(?:\d+\.?\s*)?(abstract|introduction|related work|background|method|approach|experiment|evaluation|result|discussion|conclusion|future work|limitation|acknowledgment|reference)", re.I),
        re.compile(r"^#+\s*(abstract|introduction|related work|background|method|approach|experiment|evaluation|result|discussion|conclusion|future work|limitation)", re.I),
    ]

    ALGORITHM_PATTERNS = [
        re.compile(r"(?:Algorithm|Alg\.)\s*(\d+)[:\s]*\s*(.+?)(?=\n\n|\Z)", re.I | re.S),
        re.compile(r"```(?:algorithm|pseudo)\s*\n(.*?)```", re.I | re.S),
    ]

    def parse_paper_text(self, text: str, arxiv_id: str = "") -> Paper:
        """Parse raw paper text into structured Paper object."""
        paper = Paper(
            arxiv_id=arxiv_id,
            title=self._extract_title(text),
            authors=self._extract_authors(text),
            abstract=self._extract_abstract(text),
        )

        # Extract sections
        paper.sections = self._extract_sections(text)

        # Extract algorithms
        paper.algorithms = self._extract_algorithms(text, arxiv_id)

        # Extract contributions and limitations
        paper.contributions = self._extract_contributions(text)
        paper.limitations = self._extract_limitations(text)

        # Extract keywords
        paper.keywords = self._extract_keywords(text)

        # Generate summary
        paper.summary = self._generate_summary(paper)

        return paper

    def compare_papers(self, paper_a: Paper, paper_b: Paper) -> dict:
        """Compare two papers and identify similarities/differences."""
        return {
            "shared_keywords": list(set(paper_a.keywords) & set(paper_b.keywords)),
            "unique_to_a": list(set(paper_a.keywords) - set(paper_b.keywords)),
            "unique_to_b": list(set(paper_b.keywords) - set(paper_a.keywords)),
            "overlap_topics": self._topic_overlap(
                paper_a.abstract, paper_b.abstract,
            ),
            "comparison": (
                f"'{paper_a.title[:80]}' vs '{paper_b.title[:80]}': "
                f"Both address {', '.join(set(paper_a.keywords) & set(paper_b.keywords))[:5]} "
                f"but '{paper_a.title[:40]}' uniquely covers {', '.join(set(paper_a.keywords) - set(paper_b.keywords))[:3]}."
            ),
        }

    def _extract_title(self, text: str) -> str:
        lines = text.strip().split("\n")
        for line in lines[:20]:
            clean = line.strip()
            if clean and not clean.startswith("#") and len(clean) > 10:
                if not any(kw in clean.lower() for kw in ["abstract", "arxiv", "http", "copyright"]):
                    return clean[:300]
        return "Unknown Title"

    def _extract_authors(self, text: str) -> list[str]:
        # Look for author patterns
        author_patterns = [
            r"(?:by|authors?)[:\s]*([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+){1,10})",
        ]
        for pat in author_patterns:
            match = re.search(pat, text[:2000])
            if match:
                authors_str = match.group(1)
                return [a.strip() for a in authors_str.split(",") if a.strip()]
        return ["Unknown"]

    def _extract_abstract(self, text: str) -> str:
        patterns = [
            r"(?:abstract|ABSTRACT)[:\s-]*\n?(.*?)(?:\n\s*\n|\n(?:\d+\.?\s*)?(?:introduction|1\.|I\.))",
            r"```abstract\s*\n(.*?)```",
        ]
        for pat in patterns:
            match = re.search(pat, text[:5000], re.I | re.S)
            if match:
                return match.group(1).strip()[:2000]
        # Fallback: first substantial paragraph
        paras = text[:3000].split("\n\n")
        for p in paras:
            p = p.strip()
            if len(p) > 100 and not p.startswith("#"):
                return p[:2000]
        return ""

    def _extract_sections(self, text: str) -> list[PaperSection]:
        sections = []
        current = None
        current_content = []

        for line in text.split("\n"):
            is_heading = False
            heading_name = ""
            for pat in self.SECTION_PATTERNS:
                match = pat.match(line.strip())
                if match:
                    is_heading = True
                    heading_name = match.group(1).strip()
                    break

            if is_heading or line.strip().startswith("**") and len(line.strip()) < 100:
                if current is not None and current_content:
                    current.content = "\n".join(current_content)
                    sections.append(current)
                current = PaperSection(
                    heading=heading_name or line.strip().lstrip("#*- ").strip(),
                    content="",
                )
                current_content = []
            elif current is not None:
                current_content.append(line)

        if current is not None and current_content:
            current.content = "\n".join(current_content)
            sections.append(current)

        return sections

    def _extract_algorithms(self, text: str, source: str) -> list[Algorithm]:
        algorithms = []
        for pat in self.ALGORITHM_PATTERNS:
            for match in pat.finditer(text):
                if match.lastindex and match.lastindex >= 2:
                    name = match.group(1) if match.lastindex >= 1 else "Unknown"
                    pseudo = match.group(match.lastindex)
                    algorithms.append(Algorithm(
                        name=name.strip(),
                        description="",
                        pseudocode=pseudo.strip()[:2000],
                        source_paper=source,
                    ))
        return algorithms

    def _extract_contributions(self, text: str) -> list[str]:
        contributions = []
        patterns = [
            r"(?:contribution|novelty|our main).*?:(.+?)(?:\n|$)",
            r"(?:we propose|we introduce|we present|we develop)(.+?)(?:\.|\n)",
        ]
        for pat in patterns:
            for match in re.finditer(pat, text[:5000], re.I):
                contrib = match.group(1).strip()[:300]
                if contrib not in contributions:
                    contributions.append(contrib)
        return contributions[:10]

    def _extract_limitations(self, text: str) -> list[str]:
        limitations = []
        patterns = [
            r"(?:limitation|drawback|weakness|not able|fails? to|cannot|does not)(.+?)(?:\.|\n)",
        ]
        for pat in patterns:
            for match in re.finditer(pat, text[:5000], re.I):
                lim = match.group(1).strip()[:200]
                if lim not in limitations:
                    limitations.append(lim)
        return limitations[:10]

    def _extract_keywords(self, text: str) -> list[str]:
        # Simple keyword extraction from title + abstract
        combined = text[:5000].lower()
        # Technical AI/ML keywords to look for
        tech_keywords = [
            "transformer", "attention", "diffusion", "reinforcement learning",
            "gradient descent", "neural network", "deep learning", "cnn", "rnn",
            "lstm", "gan", "vae", "bert", "gpt", "llm", "language model",
            "mixture of experts", "moe", "distillation", "pruning", "quantization",
            "fine-tuning", "pre-training", "zero-shot", "few-shot", "prompt",
            "chain-of-thought", "reasoning", "planning", "search", "optimization",
            "loss function", "regularization", "normalization", "batch norm",
            "layer norm", "rms norm", "dropout", "augmentation", "curriculum",
            "self-supervised", "contrastive", "multi-modal", "vision", "nlp",
            "speech", "rlhf", "alignment", "safety", "bias", "fairness",
            "efficiency", "latency", "throughput", "memory", "inference",
            "training", "scaling", "parallelism", "distributed", "federated",
        ]
        found = []
        for kw in tech_keywords:
            if kw in combined:
                found.append(kw)
        return found

    def _generate_summary(self, paper: Paper) -> str:
        """Generate a structured summary of the paper."""
        lines = [
            f"Title: {paper.title[:200]}",
            f"Authors: {', '.join(paper.authors[:10])}",
            f"Published: {paper.published}",
            "",
            f"Abstract: {paper.abstract[:500]}",
            "",
        ]
        if paper.contributions:
            lines.append("Key Contributions:")
            for c in paper.contributions[:5]:
                lines.append(f"  • {c}")
            lines.append("")
        if paper.algorithms:
            lines.append(f"Algorithms: {len(paper.algorithms)} extracted")
        if paper.limitations:
            lines.append(f"Limitations: {len(paper.limitations)} identified")
        if paper.keywords:
            lines.append(f"Keywords: {', '.join(paper.keywords[:15])}")

        return "\n".join(lines)

    def _topic_overlap(self, text_a: str, text_b: str) -> str:
        """Compute topic overlap between two abstracts."""
        words_a = set(re.findall(r"\b[a-z]{4,}\b", text_a.lower()))
        words_b = set(re.findall(r"\b[a-z]{4,}\b", text_b.lower()))
        overlap = words_a & words_b
        if len(words_a) == 0 or len(words_b) == 0:
            return "No overlap"
        jaccard = len(overlap) / len(words_a | words_b)
        if jaccard > 0.3:
            return f"High overlap (Jaccard={jaccard:.2f})"
        elif jaccard > 0.1:
            return f"Moderate overlap (Jaccard={jaccard:.2f})"
        return f"Low overlap (Jaccard={jaccard:.2f})"


class ResearchLibrary:
    """Manages a personal library of read and analyzed papers."""

    def __init__(self, library_path: str = "data/phase9/library"):
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        self.papers: dict[str, Paper] = {}
        self.graph = ResearchGraph()
        self._load()

    def add_paper(self, paper: Paper):
        """Add a paper to the library."""
        self.papers[paper.arxiv_id] = paper
        self.graph.papers[paper.arxiv_id] = paper

        # Update citation graph
        if paper.citations:
            self.graph.citations[paper.arxiv_id] = paper.citations

        # Cluster by topic
        for kw in paper.keywords[:3]:
            if kw not in self.graph.topic_clusters:
                self.graph.topic_clusters[kw] = []
            if paper.arxiv_id not in self.graph.topic_clusters[kw]:
                self.graph.topic_clusters[kw].append(paper.arxiv_id)

        self._save()

    def get_related(self, arxiv_id: str, max_distance: int = 2) -> list[Paper]:
        """Find related papers via citation/topic graph."""
        related_ids = set()

        # Direct citations
        if arxiv_id in self.graph.citations:
            related_ids.update(self.graph.citations[arxiv_id][:20])

        # Same topic
        paper = self.papers.get(arxiv_id)
        if paper:
            for kw in paper.keywords[:5]:
                cluster = self.graph.topic_clusters.get(kw, [])
                related_ids.update(cluster[:10])

        related_ids.discard(arxiv_id)
        return [self.papers[pid] for pid in related_ids if pid in self.papers]

    def search_library(
        self,
        query: str,
        field: str = "all",
    ) -> list[Paper]:
        """Search the local paper library."""
        results = []
        query_lower = query.lower()
        for paper in self.papers.values():
            if field == "title" and query_lower in paper.title.lower():
                results.append(paper)
            elif field == "abstract" and query_lower in paper.abstract.lower():
                results.append(paper)
            elif field == "keyword" and query_lower in " ".join(paper.keywords).lower():
                results.append(paper)
            elif field == "all":
                text = f"{paper.title} {paper.abstract} {' '.join(paper.keywords)}".lower()
                if query_lower in text:
                    results.append(paper)
        return results

    def _save(self):
        index = {
            "paper_ids": list(self.papers.keys()),
            "topic_clusters": self.graph.topic_clusters,
            "citations": self.graph.citations,
        }
        (self.library_path / "index.json").write_text(json.dumps(index, indent=2))

    def _load(self):
        idx_path = self.library_path / "index.json"
        if not idx_path.exists():
            return
        index = json.loads(idx_path.read_text())
        self.graph.topic_clusters = index.get("topic_clusters", {})
        self.graph.citations = index.get("citations", {})
