import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup

import requests
from wordpress_client import WordPressClient


class MediaWikiClient:
    def __init__(self, api_url: str, user_agent: str = "EastAsianCuisineCorpusBuilder/2.5 (coursework project)"):
        self.api_url = api_url
        self.session = requests.Session()
        self.headers = {"User-Agent": user_agent}

    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        delay = 5.0
        max_delay = 120.0

        while True:
            try:
                r = self.session.get(
                    self.api_url,
                    params=params,
                    headers=self.headers,
                    timeout=30,
                )

                if r.status_code == 429:
                    print(
                        f"[WARN] 429 Too Many Requests for {params.get('page')} | "
                        f"sleeping {delay:.1f}s and retrying until success",
                        flush=True,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, max_delay)
                    continue

                r.raise_for_status()
                data = r.json()

                if "error" in data:
                    err = data["error"]
                    code = err.get("code", "")
                    info = err.get("info", "")

                    if code in {"missingtitle", "invalidtitle"}:
                        raise ValueError(
                            f"Permanent MediaWiki error for {params.get('page')}: {code} | {info}"
                        )

                    raise requests.HTTPError(str(err))

                time.sleep(3.0)
                return data

            except ValueError:
                raise
            except requests.RequestException as e:
                print(
                    f"[WARN] request failed for {params.get('page')}: {e} | "
                    f"sleeping {delay:.1f}s and retrying until success",
                    flush=True,
                )
                time.sleep(delay)
                delay = min(delay * 2, max_delay)

    @staticmethod
    def _extract_html_from_parse_response(data: Dict[str, Any]) -> Optional[str]:
        if not isinstance(data, dict):
            return None

        parse_obj = data.get("parse")
        if not isinstance(parse_obj, dict):
            return None

        text_obj = parse_obj.get("text")
        if isinstance(text_obj, dict):
            value = text_obj.get("*")
            return value if isinstance(value, str) else None
        if isinstance(text_obj, str):
            return text_obj
        return None

    @staticmethod
    def _extract_sections_from_parse_response(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(data, dict):
            return []
        parse_obj = data.get("parse")
        if not isinstance(parse_obj, dict):
            return []
        sections = parse_obj.get("sections")
        return sections if isinstance(sections, list) else []

    def get_lead_html(self, title: str) -> Optional[str]:
        params = {
            "action": "parse",
            "page": title,
            "prop": "text",
            "section": 0,
            "format": "json",
            "redirects": 1,
        }
        return self._extract_html_from_parse_response(self._request(params))

    def get_sections(self, title: str) -> List[Dict[str, Any]]:
        params = {
            "action": "parse",
            "page": title,
            "prop": "sections",
            "format": "json",
            "redirects": 1,
        }
        return self._extract_sections_from_parse_response(self._request(params))

    def get_section_html(self, title: str, section_index: str) -> Optional[str]:
        params = {
            "action": "parse",
            "page": title,
            "section": section_index,
            "prop": "text",
            "format": "json",
            "redirects": 1,
        }
        return self._extract_html_from_parse_response(self._request(params))


BLOCKED_TITLES = [
    "references",
    "external links",
    "see also",
    "further reading",
    "bibliography",
    "gallery",
    "image gallery",
    "photo gallery",
]

CONDITIONAL_BLOCKED_TITLES = [
    "notes",
]

TITLE_GROUPS = {
    "definition": ["definition", "description", "terminology", "etymology", "name", "meaning", "what is"],
    "history": ["history", "origin", "background"],
    "composition": ["ingredients", "composition", "components", "staples", "raw materials", "local products"],
    "preparation": ["preparation", "method", "procedure", "technique", "cooking", "process", "cooking style"],
    "consumption": ["serving", "consumption", "eating", "dining", "breakfast", "etiquette", "customs"],
    "regionality": ["regional", "variations", "styles", "types", "varieties", "regional cuisines", "subcuisines"],
    "signature_items": ["dishes", "specialties", "street food", "popular dishes", "notable dishes", "signature dishes"],
    "characteristics": ["characteristics", "features", "profile", "flavor", "taste", "characteristic features"],
    "use": ["uses", "culinary use", "applications"],
    "preservation": ["preservation", "pickling", "fermentation", "drying", "salting"],
    "taxonomy": ["eight great traditions", "eight major traditions", "four great traditions", "traditions", "schools"],
}

PAGE_TYPE_GROUPS = {
    "overview": {"history", "composition", "consumption", "regionality", "signature_items", "characteristics", "preservation", "taxonomy"},
    "sub_cuisine": {"history", "composition", "preparation", "consumption", "regionality", "signature_items", "characteristics", "preservation"},
    "regional_cuisine": {"history", "composition", "preparation", "consumption", "regionality", "signature_items", "characteristics", "preservation"},
    "dish": {"definition", "history", "composition", "preparation", "consumption", "regionality", "characteristics"},
    "ingredient": {"definition", "composition", "preparation", "regionality", "use", "characteristics"},
    "method": {"definition", "preparation", "use", "regionality", "characteristics"},
    "recipe": {"definition", "composition", "preparation", "consumption", "regionality", "characteristics"},
    "recipe_index": {"definition", "composition", "regionality", "signature_items", "characteristics"},
}

PAGE_TYPE_THRESHOLDS = {
    "overview": 2,
    "sub_cuisine": 2,
    "regional_cuisine": 2,
    "dish": 1,
    "recipe": 1,
    "recipe_index": 1,
    "ingredient": 2,
    "method": 2,
}

DOMAIN_KEYWORDS = [
    "rice", "noodle", "soy", "tofu", "miso", "seaweed", "broth", "ferment", "fermented", "steam", "steamed",
    "stir", "fry", "fried", "grill", "grilled", "boil", "simmer", "roast", "roasted", "batter", "sauce", "paste",
    "dish", "cuisine", "ingredient", "ingredients", "cook", "cooked", "served", "serving", "soup", "stock",
    "dumpling", "kimchi", "ramen", "sushi", "bibimbap", "hot pot", "barbecue", "tea", "rice ball", "bun",
    "porridge", "pickled", "vinegared", "seasoned", "japanese", "chinese", "korean", "taiwanese", "hong kong",
    "macau", "mongolian", "uyghur", "tibetan", "okinawan",
]

ANSWER_PATTERN_KEYWORDS = [
    "origin", "originated", "history", "meaning", "means", "refers to", "consists of", "consist of", "made from",
    "primarily made", "prepared", "preparation", "served with", "served in", "served as", "consumed", "traditional",
    "popular", "type of", "style of", "dish of", "variety of", "staple", "side dish", "broth", "paste",
    "soup stock", "wrapped", "filled", "grilled", "fried", "steamed", "fermented", "pickled", "vinegared",
    "seasoned", "multi-course", "fusion", "hot pot", "barbecue", "small side dishes", "literally means",
]

WORDPRESS_EAST_ASIAN_KEYWORDS = [
    "japan", "japanese", "korea", "korean", "china", "chinese", "taiwan", "taiwanese",
    "hong kong", "macau", "mongolia", "mongolian", "east asian", "ramen", "sushi", "kimchi",
]

NOISE_MARKERS = [
    "retrieved", "archived", "isbn", "issn", "doi", "wayback machine",
    "external links", "further reading", "bibliography", "citation needed",
]

TRUNCATE_MARKERS = [
    "references",
    "external links",
    "further reading",
    "bibliography",
    "notes",
]

NOISY_TABLE_CLASSES = {
    "navbox",
    "vertical-navbox",
    "metadata",
    "mbox-small",
    "plainlinks",
    "infobox",
    "sidebar",
    "wikitable mw-collapsible",
}

DOM_NOISE_SELECTORS = [
    "style",
    "script",
    ".mw-editsection",
    ".reflist",
    ".references",
    ".mw-references-wrap",
    "div.reflist",
    "div.references",
    "ol.references",
    "sup.reference",
    "span.mw-ref",
    ".reference",
    ".thumbcaption .reference",
]

SHORT_DEFINITION_PATTERNS = [
    r"\bis an?\b",
    r"\brefers to\b",
    r"\bconsists? of\b",
    r"\bmeans\b",
    r"\bliterally means\b",
    r"\bmade from\b",
    r"\bserved with\b",
    r"\boriginated in\b",
    r"\ba type of\b",
    r"\ba variety of\b",
]

LEAD_NOISE_PATTERNS = [
    r"redirects here",
    r"for other uses, see",
    r"this article is about",
    r"may refer to",
]


TARGET_CUISINE_TERMS = [
    "east asian",
    "chinese",
    "japanese",
    "korean",
    "taiwanese",
    "hong kong",
    "macanese",
    "okinawan",
    "hakka",
    "sichuan",
    "cantonese",
    "fujian",
    "hunan",
    "jiangsu",
    "shandong",
    "zhejiang",
    "anhui",
    "uyghur",
    "mongolian",
    "tibetan",
]

TARGET_SECTION_SIGNAL_TERMS = [
    "flavor",
    "taste",
    "ingredients",
    "staple",
    "dish",
    "specialty",
    "specialties",
    "technique",
    "cooking",
    "preservation",
    "ferment",
    "pickled",
    "dining",
    "customs",
    "street food",
    "regional",
    "tradition",
    "traditions",
    "subcuisine",
    "subcuisines",
]

LIST_STYLE_KNOWLEDGE_PATTERNS = [
    r"\beight (major|great)\b",
    r"\bfour (major|great)\b",
    r"\bknown for\b",
    r"\bconsists? of\b",
    r"\boriginates? in\b",
    r"\buses\b",
]

REGIONAL_CUISINE_HINTS = [
    "cuisine",
    "food",
    "street food",
    "culinary",
]

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_for_hash(text).encode("utf-8")).hexdigest()


def tokenize_for_similarity(text: str) -> Set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_for_hash(text)))


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")
    normalized: List[str] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            normalized.append("")
            continue

        is_list_like = bool(re.match(r"^(\d+\.)|^[-*]\s", line))
        if is_list_like:
            normalized.append(line)
            continue

        if normalized and normalized[-1] != "":
            normalized[-1] = normalized[-1] + " " + line
        else:
            normalized.append(line)

    text = "\n".join(normalized)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def is_heading_line(line: str) -> bool:
    candidate = normalize(line)
    return candidate in TRUNCATE_MARKERS


def should_drop_table(tag: Any) -> bool:
    if tag is None:
        return True

    attrs = getattr(tag, "attrs", None) or {}
    if not isinstance(attrs, dict):
        attrs = {}

    raw_classes = attrs.get("class") or []
    if isinstance(raw_classes, str):
        class_list = [raw_classes]
    elif isinstance(raw_classes, (list, tuple, set)):
        class_list = [str(x) for x in raw_classes if x is not None]
    else:
        class_list = [str(raw_classes)] if raw_classes else []

    classes = set(class_list)
    if classes & NOISY_TABLE_CLASSES:
        return True

    role = attrs.get("role") or ""
    elem_id = attrs.get("id") or ""
    attrs_text = " ".join([str(role), " ".join(class_list), str(elem_id)]).lower()
    noisy_terms = ["navbox", "infobox", "metadata", "reference", "sidebar"]
    return any(term in attrs_text for term in noisy_terms)


def flatten_table_to_text(table: Any) -> str:
    if table is None or not hasattr(table, "find_all"):
        return ""
    rows: List[str] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        cleaned = [cell.get_text(" ", strip=True) for cell in cells if hasattr(cell, "get_text")]
        cleaned = [c for c in cleaned if c]
        if cleaned:
            rows.append(" | ".join(cleaned))
    return "\n".join(rows).strip()


def clean_inline_noise(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = re.sub(r"\[\d+\]", "", line)
    line = re.sub(r"\[citation needed\]", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\(\s*citation needed\s*\)", "", line, flags=re.IGNORECASE)
    line = re.sub(r"(?i)\s+\^\s*$", "", line)
    line = re.sub(r"\s{2,}", " ", line).strip()
    return line


def is_reference_line(line: str) -> bool:
    line_n = normalize(line)
    if not line_n:
        return False
    hard_patterns = [
        r"^retrieved\b",
        r"^archived\b",
        r"^isbn\b",
        r"^issn\b",
        r"^doi\b",
        r"^wayback machine\b",
        r"^cite error\b",
    ]
    if any(re.search(pat, line_n) for pat in hard_patterns):
        return True

    citation_like = 0
    citation_like += int("retrieved" in line_n)
    citation_like += int("doi" in line_n)
    citation_like += int("isbn" in line_n)
    citation_like += int("issn" in line_n)
    citation_like += int("archive" in line_n)
    citation_like += int("wayback machine" in line_n)
    return citation_like >= 2


def clean_html_to_text(html: str) -> str:
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for selector in DOM_NOISE_SELECTORS:
        for tag in soup.select(selector):
            try:
                tag.decompose()
            except Exception:
                pass

    for tag in soup.find_all(id=re.compile(r"cite_note-", re.IGNORECASE)):
        try:
            tag.decompose()
        except Exception:
            pass

    for table in soup.find_all("table"):
        try:
            if should_drop_table(table):
                table.decompose()
                continue
            flattened = flatten_table_to_text(table)
            if flattened:
                table.replace_with("\n" + flattened + "\n")
            else:
                table.decompose()
        except Exception:
            continue

    text = soup.get_text("\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = clean_inline_noise(raw_line)
        if not line:
            cleaned_lines.append("")
            continue
        if is_heading_line(line):
            break
        if re.match(r"(?im)^main article:\s*.*$", line):
            continue
        if re.match(r"(?im)^see also:\s*.*$", line):
            continue
        if re.match(r"(?im)^for other uses, see\b.*$", line):
            continue
        if re.match(r"(?im)^.*redirects here\.?$", line):
            continue
        if is_reference_line(line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = normalize_newlines(text)
    return text.strip()


def looks_like_noise(text: str) -> bool:
    text_n = normalize(text)
    words = text.split()

    if len(words) <= 4:
        return True

    marker_hits = sum(1 for marker in NOISE_MARKERS if marker in text_n)
    if marker_hits >= 2:
        return True

    if re.fullmatch(r"[\W\d_ ]+", text):
        return True

    return False


def count_keyword_hits(text_n: str, keywords: List[str]) -> int:
    return sum(1 for kw in keywords if kw in text_n)


def matched_title_groups(page_type: str, section_title: str) -> List[str]:
    title = normalize(section_title)
    allowed_groups = PAGE_TYPE_GROUPS.get(page_type, set())
    matched: List[str] = []
    for group_name in allowed_groups:
        terms = TITLE_GROUPS.get(group_name, [])
        if any(term in title for term in terms):
            matched.append(group_name)
    return matched


def has_short_definition_pattern(text: str) -> bool:
    text_n = normalize(text)
    return any(re.search(pat, text_n) for pat in SHORT_DEFINITION_PATTERNS)


def is_lead_boilerplate(text: str) -> bool:
    text_n = normalize(text)
    return any(re.search(pat, text_n) for pat in LEAD_NOISE_PATTERNS)


def normalize_page_type(page_type: str, title: str = "") -> str:
    page_type_n = normalize(page_type)
    title_n = normalize(title)
    if page_type_n in PAGE_TYPE_GROUPS:
        return page_type_n
    if page_type_n == "sub_cuisine":
        return "sub_cuisine"
    if "cuisine" in title_n and any(term in title_n for term in TARGET_CUISINE_TERMS):
        return "regional_cuisine"
    return page_type_n or "overview"


def is_target_cuisine_page(title: str, text: str = "") -> bool:
    haystack = normalize(title + " " + text[:500])
    return any(term in haystack for term in TARGET_CUISINE_TERMS)


def count_target_signal_hits(section_title: str, text: str) -> int:
    haystack = normalize(section_title + " " + text[:1200])
    return sum(1 for term in TARGET_SECTION_SIGNAL_TERMS if term in haystack)


def has_list_style_knowledge(text: str) -> bool:
    text_n = normalize(text)
    list_like = len(re.findall(r"(^|\n)\s*(?:[-*]|\d+\.)\s+", text)) >= 2
    pattern_hit = any(re.search(pat, text_n) for pat in LIST_STYLE_KNOWLEDGE_PATTERNS)
    return list_like or pattern_hit


def should_preblock_title(section_title: str, page_type: str) -> Optional[str]:
    title_n = normalize(section_title)
    normalized_page_type = normalize_page_type(page_type)
    if any(blocked in title_n for blocked in BLOCKED_TITLES):
        return "blocked_title_precheck"
    if any(blocked in title_n for blocked in CONDITIONAL_BLOCKED_TITLES):
        if normalized_page_type not in {"dish", "recipe", "recipe_index"}:
            return "blocked_title_precheck"
    return None



def should_keep_short_text(text: str, page_type: str, title_group_hits: List[str], domain_hits: int, answer_hits: int, target_signal_hits: int, is_target_page: bool) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    wc = len(text.split())
    if wc >= 20:
        return True, reasons

    if has_short_definition_pattern(text):
        reasons.append("short_definition_pattern")
        return True, reasons

    if page_type in {"dish", "recipe", "recipe_index"} and (title_group_hits or domain_hits >= 1 or answer_hits >= 1):
        reasons.append("short_recipe_or_dish_allowed")
        return True, reasons

    if is_target_page and target_signal_hits >= 2 and (domain_hits >= 1 or answer_hits >= 1):
        reasons.append("short_target_cuisine_section_allowed")
        return True, reasons

    if answer_hits >= 1 and domain_hits >= 1:
        reasons.append("short_answer_pattern_supported")
        return True, reasons

    return False, reasons


def score_section(page_type: str, section_title: str, text: str, is_lead: bool = False, page_title: str = "") -> Tuple[int, bool, List[str]]:
    normalized_page_type = normalize_page_type(page_type, page_title)
    title = normalize(section_title)
    text_n = normalize(text)
    reasons: List[str] = []
    score = 0

    if any(b in title for b in BLOCKED_TITLES):
        reasons.append("blocked_title")
        return -999, False, reasons

    if not text.strip():
        reasons.append("empty_text")
        return -999, False, reasons

    if looks_like_noise(text):
        reasons.append("noise_text")
        return -999, False, reasons

    if is_lead and is_lead_boilerplate(text):
        reasons.append("lead_boilerplate")
        return -999, False, reasons

    wc = len(text.split())
    threshold = PAGE_TYPE_THRESHOLDS.get(normalized_page_type, 2)

    if is_lead:
        score += 1
        reasons.append("lead_section")

    title_group_hits = matched_title_groups(normalized_page_type, section_title)
    if title_group_hits:
        title_group_score = min(3, len(title_group_hits))
        if "taxonomy" in title_group_hits and normalized_page_type in {"overview", "sub_cuisine", "regional_cuisine"}:
            title_group_score += 1
            reasons.append("taxonomy_title_boost")
        score += title_group_score
        reasons.append(f"title_groups={','.join(title_group_hits)}")
    elif not is_lead:
        reasons.append("no_title_group_match")

    domain_hits = count_keyword_hits(text_n, DOMAIN_KEYWORDS)
    answer_hits = count_keyword_hits(text_n, ANSWER_PATTERN_KEYWORDS)
    target_signal_hits = count_target_signal_hits(section_title, text)
    is_target_page = is_target_cuisine_page(page_title or section_title, text)

    if domain_hits:
        score += min(2, domain_hits)
        reasons.append(f"domain_hits={domain_hits}")
    if answer_hits:
        score += min(2, answer_hits)
        reasons.append(f"answer_hits={answer_hits}")
    if target_signal_hits:
        score += min(2, target_signal_hits)
        reasons.append(f"target_signal_hits={target_signal_hits}")

    if has_short_definition_pattern(text):
        score += 2
        reasons.append("definition_pattern_boost")

    if is_target_page and target_signal_hits >= 2 and normalized_page_type in {"overview", "sub_cuisine", "regional_cuisine"}:
        score += 1
        reasons.append("target_cuisine_overview_boost")

    if has_list_style_knowledge(text) and normalized_page_type in {"overview", "sub_cuisine", "regional_cuisine"}:
        score += 1
        reasons.append("list_style_knowledge_boost")

    if wc > 80:
        score += 1
        reasons.append("content_substantial")
    elif wc >= 30:
        score += 1
        reasons.append("content_moderate")
    else:
        reasons.append("short_text")
        short_keep, short_reasons = should_keep_short_text(
            text, normalized_page_type, title_group_hits, domain_hits, answer_hits, target_signal_hits, is_target_page
        )
        reasons.extend(short_reasons)
        if not short_keep:
            return score, False, reasons

    keep = False
    if is_lead:
        keep = score >= 1 or domain_hits >= 1 or answer_hits >= 1 or target_signal_hits >= 2
    elif score >= threshold:
        keep = True
    elif normalized_page_type in {"dish", "recipe", "recipe_index"} and (title_group_hits or domain_hits >= 1 or answer_hits >= 1):
        keep = True
        reasons.append("page_type_recall_override")
    elif is_target_page and normalized_page_type in {"overview", "sub_cuisine", "regional_cuisine"} and target_signal_hits >= 2 and (domain_hits >= 1 or answer_hits >= 1 or has_list_style_knowledge(text)):
        keep = True
        reasons.append("target_cuisine_recall_override")
    elif wc >= 20 and answer_hits >= 1 and domain_hits >= 1:
        keep = True
        reasons.append("qa_pattern_recall_override")

    return score, keep, reasons


def chunk_text(text: str, target_words: int = 160, min_words: int = 60, overlap_words: int = 25) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for para in paragraphs:
        pw = len(para.split())
        if current_words and current_words + pw > target_words:
            combined = "\n\n".join(current).strip()
            if len(combined.split()) >= min_words:
                chunks.append(combined)
            if overlap_words > 0 and combined:
                words = combined.split()
                overlap = " ".join(words[-overlap_words:])
                current = [overlap, para]
                current_words = len(overlap.split()) + pw
            else:
                current = [para]
                current_words = pw
        else:
            current.append(para)
            current_words += pw

    if current:
        combined = "\n\n".join(current).strip()
        if len(combined.split()) >= min_words or not chunks:
            chunks.append(combined)
    return [c for c in chunks if not looks_like_noise(c)]


def deduplicate_records(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, str]]]:
    deduped: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    duplicate_log: List[Dict[str, str]] = []

    for rec in records:
        text = rec.get("text", "")
        h = text_hash(text)
        if h in seen:
            duplicate_log.append({
                "level": "document",
                "duplicate_doc_id": rec["doc_id"],
                "kept_doc_id": seen[h],
            })
            continue
        seen[h] = rec["doc_id"]
        deduped.append(rec)

    stats = {
        "before": len(records),
        "after": len(deduped),
        "removed": len(records) - len(deduped),
    }
    return deduped, stats, duplicate_log


def deduplicate_chunks(
    chunks: List[Dict[str, Any]],
    exact_threshold: float = 1.0,
    near_dup_threshold: float = 0.88,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[Dict[str, Any]]]:
    deduped: List[Dict[str, Any]] = []
    exact_seen: Dict[str, str] = {}
    duplicate_log: List[Dict[str, Any]] = []

    for chunk in chunks:
        text = chunk.get("chunk_text", "")
        h = text_hash(text)
        if h in exact_seen:
            duplicate_log.append({
                "level": "chunk_exact",
                "duplicate_chunk_id": chunk["chunk_id"],
                "kept_chunk_id": exact_seen[h],
            })
            continue

        tokens = tokenize_for_similarity(text)
        duplicate_of: Optional[str] = None
        duplicate_score = 0.0

        for kept in deduped:
            sim = jaccard_similarity(tokens, tokenize_for_similarity(kept["chunk_text"]))
            if sim >= exact_threshold:
                duplicate_of = kept["chunk_id"]
                duplicate_score = sim
                break
            if sim >= near_dup_threshold:
                duplicate_of = kept["chunk_id"]
                duplicate_score = sim
                break

        if duplicate_of is not None:
            duplicate_log.append({
                "level": "chunk_near",
                "duplicate_chunk_id": chunk["chunk_id"],
                "kept_chunk_id": duplicate_of,
                "similarity": round(duplicate_score, 4),
            })
            continue

        exact_seen[h] = chunk["chunk_id"]
        deduped.append(chunk)

    stats = {
        "before": len(chunks),
        "after": len(deduped),
        "removed": len(chunks) - len(deduped),
    }
    return deduped, stats, duplicate_log


def build_mediawiki_url(api_url: str, title: str) -> str:
    if "en.wikipedia.org" in api_url:
        return f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    if "en.wikibooks.org" in api_url:
        return f"https://en.wikibooks.org/wiki/{title.replace(' ', '_')}"
    return ""


def is_permanent_page_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "permanent mediawiki error" in message or "missingtitle" in message or "invalidtitle" in message


def process_mediawiki_source(source_cfg: Dict[str, Any], out_records: List[Dict[str, Any]], out_chunks: List[Dict[str, Any]], scoring_log: List[Dict[str, Any]]) -> Dict[str, int]:
    client = MediaWikiClient(source_cfg["api_url"])
    source_name = source_cfg["name"]
    counts = {"pages": 0, "records": 0, "chunks": 0, "failures": 0}

    seeds = source_cfg.get("pages", []) or source_cfg.get("seeds", [])

    for seed in seeds:
        title = seed["title"]
        page_type = seed["page_type"]
        print(f"[INFO] {source_name}: {title}", flush=True)
        counts["pages"] += 1

        page_url = seed.get("url") or build_mediawiki_url(source_cfg["api_url"], title)

        try:
            lead_html = client.get_lead_html(title)
            if not lead_html:
                scoring_log.append({
                    "source": source_name,
                    "title": title,
                    "section": "Lead",
                    "score": -999,
                    "keep": False,
                    "reasons": ["empty_or_missing_lead_html"],
                })
                print(f"[DROP] {title} -> Lead | score=-999 | empty_or_missing_lead_html", flush=True)
            else:
                try:
                    lead_text = clean_html_to_text(lead_html)
                except Exception as clean_exc:
                    raise RuntimeError(f"lead_cleaning_failed: {clean_exc}") from clean_exc

                score, keep, reasons = score_section(page_type, "Lead", lead_text, is_lead=True, page_title=title)
                scoring_log.append({"source": source_name, "title": title, "section": "Lead", "score": score, "keep": keep, "reasons": reasons})
                if keep:
                    doc_id = f"{source_name.lower().replace(' ', '_')}_{title.lower().replace(' ', '_')}_lead"
                    rec = {
                        "doc_id": doc_id,
                        "source": source_name,
                        "page_type": page_type,
                        "title": title,
                        "section": "Lead",
                        "url": page_url,
                        "cuisine": "East Asian",
                        "text": lead_text,
                    }
                    out_records.append(rec)
                    counts["records"] += 1
                    for idx, chunk in enumerate(chunk_text(lead_text), start=1):
                        out_chunks.append({**rec, "chunk_id": f"{doc_id}_chunk_{idx:02d}", "chunk_text": chunk})
                        counts["chunks"] += 1
                else:
                    print(f"[DROP] {title} -> Lead | score={score} | {';'.join(reasons)}", flush=True)
        except Exception as e:
            counts["failures"] += 1
            if is_permanent_page_error(e):
                print(f"[SKIP] lead skipped for missing/invalid page {title}: {e}", flush=True)
            else:
                print(f"[WARN] lead failed for {title}: {type(e).__name__}: {e}", flush=True)

        try:
            sections = client.get_sections(title)
            print(f"[INFO] {title}: {len(sections)} sections found", flush=True)
        except Exception as e:
            counts["failures"] += 1
            if is_permanent_page_error(e):
                print(f"[SKIP] sections skipped for missing/invalid page {title}: {e}", flush=True)
            else:
                print(f"[WARN] sections failed for {title}: {e}", flush=True)
            continue

        for sec in sections:
            section_title = sec.get("line", "")
            section_index = sec.get("index", "")
            if not section_title or not section_index:
                continue

            preblock_reason = should_preblock_title(section_title, page_type)
            if preblock_reason is not None:
                scoring_log.append({
                    "source": source_name,
                    "title": title,
                    "section": section_title,
                    "score": -999,
                    "keep": False,
                    "reasons": [preblock_reason],
                })
                print(f"[DROP] {title} -> {section_title} | score=-999 | {preblock_reason}", flush=True)
                continue

            try:
                html = client.get_section_html(title, section_index)
                if not html:
                    scoring_log.append({
                        "source": source_name,
                        "title": title,
                        "section": section_title,
                        "score": -999,
                        "keep": False,
                        "reasons": ["empty_or_missing_section_html"],
                    })
                    print(f"[DROP] {title} -> {section_title} | score=-999 | empty_or_missing_section_html", flush=True)
                    continue
                text = clean_html_to_text(html)
                score, keep, reasons = score_section(page_type, section_title, text, is_lead=False, page_title=title)
                scoring_log.append({"source": source_name, "title": title, "section": section_title, "score": score, "keep": keep, "reasons": reasons})
                print(f"[{'KEEP' if keep else 'DROP'}] {title} -> {section_title} | score={score} | {';'.join(reasons)}", flush=True)
                if not keep:
                    continue
                doc_id = f"{source_name.lower().replace(' ', '_')}_{title.lower().replace(' ', '_')}_{section_index}"
                rec = {
                    "doc_id": doc_id,
                    "source": source_name,
                    "page_type": page_type,
                    "title": title,
                    "section": section_title,
                    "url": page_url,
                    "cuisine": "East Asian",
                    "text": text,
                }
                out_records.append(rec)
                counts["records"] += 1
                for idx, chunk in enumerate(chunk_text(text), start=1):
                    out_chunks.append({**rec, "chunk_id": f"{doc_id}_chunk_{idx:02d}", "chunk_text": chunk})
                    counts["chunks"] += 1
                time.sleep(0.2)
            except Exception as e:
                counts["failures"] += 1
                if is_permanent_page_error(e):
                    print(f"[SKIP] section skipped for missing/invalid page {title}::{section_title}: {e}", flush=True)
                else:
                    print(f"[WARN] section failed for {title}::{section_title}: {e}", flush=True)
    return counts


def process_wordpress_source(source_cfg: Dict[str, Any], out_records: List[Dict[str, Any]], out_chunks: List[Dict[str, Any]], scoring_log: List[Dict[str, Any]]) -> Dict[str, int]:
    client = WordPressClient(source_cfg["base_url"])
    source_name = source_cfg["name"]
    counts = {"pages": 0, "records": 0, "chunks": 0, "failures": 0}
    posts: List[Dict[str, Any]] = []

    try:
        posts.extend(client.get_posts(per_page=50, page=1))
    except Exception as e:
        counts["failures"] += 1
        print(f"[WARN] posts failed: {e}", flush=True)
    try:
        posts.extend(client.get_pages(per_page=50, page=1))
    except Exception as e:
        counts["failures"] += 1
        print(f"[WARN] pages failed: {e}", flush=True)

    for item in posts:
        title = BeautifulSoup(item.get("title", {}).get("rendered", ""), "html.parser").get_text(" ").strip()
        html = item.get("content", {}).get("rendered", "")
        if not title or not html:
            continue
        text = clean_html_to_text(html)
        haystack = normalize(title + "\n" + text)
        if not any(k in haystack for k in WORDPRESS_EAST_ASIAN_KEYWORDS):
            continue
        counts["pages"] += 1
        score, keep, reasons = score_section("overview", title, text, is_lead=False, page_title=title)
        reasons.append("wordpress_post")
        scoring_log.append({"source": source_name, "title": title, "section": "full_post", "score": score, "keep": keep, "reasons": reasons})
        print(f"[{'KEEP' if keep else 'DROP'}] {source_name} -> {title} | score={score}", flush=True)
        if not keep:
            continue
        doc_id = f"{source_name.lower().replace(' ', '_')}_{item.get('id', 'post')}"
        rec = {
            "doc_id": doc_id,
            "source": source_name,
            "page_type": "overview",
            "title": title,
            "section": "full_post",
            "url": item.get("link", ""),
            "cuisine": "East Asian",
            "text": text,
        }
        out_records.append(rec)
        counts["records"] += 1
        for idx, chunk in enumerate(chunk_text(text), start=1):
            out_chunks.append({**rec, "chunk_id": f"{doc_id}_chunk_{idx:02d}", "chunk_text": chunk})
            counts["chunks"] += 1
    return counts


def write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--outdir", default="output")
    args = parser.parse_args()

    with open(args.seeds, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_records: List[Dict[str, Any]] = []
    all_chunks: List[Dict[str, Any]] = []
    scoring_log: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {"sources": {}}

    print("[1/5] Loading seeds...", flush=True)
    for source_cfg in cfg.get("sources", []):
        source_name = source_cfg["name"]
        print(f"\n[2/5] Processing source: {source_name}", flush=True)
        if source_cfg["type"] == "mediawiki":
            counts = process_mediawiki_source(source_cfg, all_records, all_chunks, scoring_log)
        elif source_cfg["type"] == "wordpress":
            counts = process_wordpress_source(source_cfg, all_records, all_chunks, scoring_log)
        else:
            print(f"[WARN] Unknown source type: {source_cfg['type']}", flush=True)
            continue
        summary["sources"][source_name] = counts

    print("\n[3/5] Deduplicating documents...", flush=True)
    all_records, doc_dedup_stats, doc_duplicate_log = deduplicate_records(all_records)

    kept_doc_ids = {rec["doc_id"] for rec in all_records}
    all_chunks = [chunk for chunk in all_chunks if chunk["doc_id"] in kept_doc_ids]

    print("[4/5] Deduplicating chunks...", flush=True)
    all_chunks, chunk_dedup_stats, chunk_duplicate_log = deduplicate_chunks(all_chunks)

    print("[5/5] Writing files...", flush=True)
    write_jsonl(outdir / "east_asian_corpus_v9_singlefile.jsonl", all_records)
    write_jsonl(outdir / "east_asian_chunks_v9_singlefile.jsonl", all_chunks)
    write_jsonl(outdir / "section_scoring_log_v9_singlefile.jsonl", scoring_log)
    write_jsonl(outdir / "duplicate_documents_v9_singlefile.jsonl", doc_duplicate_log)
    write_jsonl(outdir / "duplicate_chunks_v9_singlefile.jsonl", chunk_duplicate_log)

    with (outdir / "metadata_v9_singlefile.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "source", "page_type", "title", "section", "url", "cuisine"])
        writer.writeheader()
        for rec in all_records:
            writer.writerow({k: rec.get(k, "") for k in ["doc_id", "source", "page_type", "title", "section", "url", "cuisine"]})

    summary["deduplication"] = {
        "documents": doc_dedup_stats,
        "chunks": chunk_dedup_stats,
    }
    summary["total_records"] = len(all_records)
    summary["total_chunks"] = len(all_chunks)
    summary["total_scored_sections"] = len(scoring_log)
    with (outdir / "summary_v9_singlefile.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE]", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
