"""
Output parser for Titan model responses.

Handles:
- ### FILE: path markers (primary format)
- --- separators between files
- <think> blocks (stripped from final output)
- Code fence extraction
"""

import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedFile:
    path: str
    content: str
    language: str = "python"


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def extract_code_from_fences(text: str) -> str:
    """Extract code content from fenced code blocks (```lang ... ```)."""
    match = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_backend_response(raw: str) -> list[ParsedFile]:
    """
    Parse a single-file backend response.

    Backend generates one file per turn, so we expect:
    - Optional <think> block
    - ### FILE: path/to/file.py
    - ```python ... ```
    """
    text = strip_think_blocks(raw)
    if not text:
        return []

    # Try ### FILE: header
    file_match = re.search(r"###\s*FILE:\s*(.+?)(?:\n|$)", text)

    if file_match:
        path = file_match.group(1).strip()
        # Get everything after the header
        after_header = text[file_match.end():]
        content = extract_code_from_fences(after_header)
    else:
        # No header — treat entire output as one file
        path = "unknown.py"
        content = extract_code_from_fences(text)

    if not content:
        return []

    # Clean up path
    path = path.strip("`").strip('"').strip("'")
    language = "python" if path.endswith(".py") else "typescript"

    return [ParsedFile(path=path, content=content, language=language)]


def parse_frontend_response(raw: str) -> list[ParsedFile]:
    """
    Parse a multi-file frontend response.

    Frontend generates all 6 files in one response:
    - ### FILE: types.ts
    - ```typescript ... ```
    - ---
    - ### FILE: composables/useLead.ts
    - ...
    """
    text = strip_think_blocks(raw)
    if not text:
        return []

    files = []

    # Split by ### FILE: markers
    pattern = re.compile(r"###\s*FILE:\s*(.+?)(?:\n|$)", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        # Try splitting by --- separators
        return _parse_by_separator(text)

    for i, match in enumerate(matches):
        path = match.group(1).strip().strip("`").strip('"').strip("'")

        # Get text between this match and the next (or end)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end]

        # Remove trailing --- separator
        chunk = re.sub(r"\n---\s*$", "", chunk.strip())

        content = extract_code_from_fences(chunk)
        if content:
            language = _detect_language(path)
            # Remove leading path duplicates (e.g. modules/crm/ prefix already in path)
            path = _clean_frontend_path(path)
            files.append(ParsedFile(path=path, content=content, language=language))

    logger.info(f"Parsed {len(files)} frontend files")
    return files


def parse_debug_response(raw: str) -> list[ParsedFile]:
    """
    Parse a debug fix response from Groq.

    Same format as frontend (### FILE: + code blocks),
    but may contain backend Python or frontend Vue/TS files.
    """
    text = strip_think_blocks(raw)
    if not text:
        return []

    files = []
    pattern = re.compile(r"###\s*FILE:\s*(.+?)(?:\n|$)", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        # Try finding any code block
        content = extract_code_from_fences(text)
        if content:
            files.append(ParsedFile(path="fix.py", content=content))
        return files

    for i, match in enumerate(matches):
        path = match.group(1).strip().strip("`").strip('"').strip("'")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].strip()
        chunk = re.sub(r"\n---\s*$", "", chunk)

        content = extract_code_from_fences(chunk)
        if content:
            language = _detect_language(path)
            files.append(ParsedFile(path=path, content=content, language=language))

    return files


def _parse_by_separator(text: str) -> list[ParsedFile]:
    """Fallback: split by --- separators and try to detect file headers."""
    chunks = re.split(r"\n---\n", text)
    files = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        # Try to find a file path in the first lines
        path = "unknown.ts"
        first_line = chunk.split("\n")[0]
        if re.match(r"^[\w/\[\]._-]+\.\w+", first_line):
            path = first_line.strip()
            chunk = "\n".join(chunk.split("\n")[1:])

        content = extract_code_from_fences(chunk)
        if content and len(content) > 10:
            language = _detect_language(path)
            files.append(ParsedFile(path=path, content=content, language=language))

    return files


def _detect_language(path: str) -> str:
    if path.endswith(".py"):
        return "python"
    if path.endswith(".vue"):
        return "vue"
    if path.endswith(".ts"):
        return "typescript"
    return "text"


def _clean_frontend_path(path: str) -> str:
    """
    Fix double path prefixes from frontend output.

    The model outputs paths like: modules/crm/pages/lead/index.vue
    But the agent also prepends: frontend/modules/crm/
    This causes: frontend/modules/crm/modules/crm/pages/...

    Solution: strip leading modules/<name>/ from the model output.
    """
    # Remove leading "modules/<anything>/" if present
    cleaned = re.sub(r"^modules/[^/]+/", "", path)
    return cleaned
