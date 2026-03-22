"""
Output parser for Titan model responses.

Handles:
- ### FILE: path markers (primary format, used by backend)
- --- separators between files (primary format, used by frontend)
- <think> blocks (stripped from final output)
- Code fence extraction
- Content-based file detection for frontend (types.ts, composable, vue pages)
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

    The Titan-UI model was trained with TWO possible formats:

    Format A (### FILE: markers):
      ### FILE: types.ts
      ```typescript ... ```
      ---
      ### FILE: composables/useLead.ts
      ...

    Format B (--- separators only, NO file headers):
      <think>...</think>
      import { z } from 'zod'
      export interface Lead { ... }
      ---
      import type { Lead } from '~/modules/crm/types'
      export function useCrm() { ... }
      ---
      <script setup lang="ts">...</script>
      <template>...</template>
      ---
      ...

    Both formats are handled.
    """
    text = strip_think_blocks(raw)
    if not text:
        return []

    files = []

    # Try ### FILE: markers first (Format A)
    pattern = re.compile(r"###\s*FILE:\s*(.+?)(?:\n|$)", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if matches:
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
                path = _clean_frontend_path(path)
                files.append(ParsedFile(path=path, content=content, language=language))

        logger.info(f"Parsed {len(files)} frontend files (### FILE: format)")
        return files

    # Format B: Split by --- separators (the model's primary trained format)
    files = _parse_by_separator_smart(text)
    logger.info(f"Parsed {len(files)} frontend files (--- separator format)")
    return files


def _parse_by_separator_smart(text: str) -> list[ParsedFile]:
    """
    Parse frontend output split by --- separators.

    Uses content-based detection to determine what each chunk is:
    1. TypeScript with 'export interface' → types.ts
    2. TypeScript with 'export function use' → composable
    3. Vue with DataTable / list / index → list page
    4. Vue with form / @submit → create/form page
    5. Vue with route.params / detail → detail page
    6. Vue with onMounted + fetch by ID → edit page

    The expected order from training data is:
    1. types.ts
    2. composables/use{Entity}.ts
    3. pages/{entity}/index.vue (list)
    4. pages/{entity}/create.vue (form)
    5. pages/{entity}/[id]/edit.vue (edit)
    6. pages/{entity}/[id]/index.vue (detail)
    """
    # Split by --- (on its own line)
    chunks = re.split(r"\n---\n", text)
    files = []

    # Detect entity name from the first chunk (types.ts)
    entity_name = None
    entity_lower = None

    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk or len(chunk) < 20:
            continue

        # Check if this chunk has code fences
        content = extract_code_from_fences(chunk)
        if not content or len(content) < 10:
            continue

        file_type = _detect_file_type(content, i)

        # Extract entity name from types.ts (first file)
        if file_type == "types" and entity_name is None:
            entity_match = re.search(r"export\s+interface\s+(\w+)", content)
            if entity_match:
                entity_name = entity_match.group(1)
                entity_lower = entity_name[0].lower() + entity_name[1:]
                # Convert CamelCase to kebab-case for paths
                entity_lower = re.sub(r'(?<!^)(?=[A-Z])', '-', entity_lower).lower()
                # Also try simple lowercase
                if '-' not in entity_lower:
                    entity_lower = entity_name.lower()

        # Assign path and language based on detected type
        path, language = _assign_path(file_type, entity_name, entity_lower, i)

        files.append(ParsedFile(path=path, content=content, language=language))

    return files


def _detect_file_type(content: str, index: int) -> str:
    """Detect what type of frontend file this content represents."""

    has_vue_template = bool(re.search(r"<template>|<template\s", content))
    has_vue_script = bool(re.search(r"<script\s+setup", content))
    is_vue = has_vue_template or has_vue_script

    if not is_vue:
        # TypeScript file
        if "export interface" in content or "export type " in content:
            if "z.object" in content or "z.string" in content:
                # Has both interface and Zod schema → types.ts
                return "types"
            return "types"
        if "export function use" in content or "export const use" in content:
            return "composable"
        # Default TS
        return "types" if index == 0 else "composable"

    # Vue file — determine which page type
    content_lower = content.lower()

    # List page indicators
    is_list = (
        "DataTable" in content
        or "datatable" in content_lower
        or ("paginator" in content_lower and "rows" in content_lower)
        or "fetchAll" in content
        or "fetchApproval" in content  # fetchXxxs pattern
    )

    # Form/Create page indicators
    is_form = (
        "@submit" in content
        or "onSubmit" in content
        or ("formData" in content and "validate" in content)
        or re.search(r"emit\s*\(\s*['\"]submit", content)
    )

    # Edit page indicators (has route params AND pre-fills form)
    is_edit = (
        ("route.params" in content or "useRoute" in content)
        and ("update" in content_lower or "edit" in content_lower or "onMounted" in content)
        and is_form
    )

    # Detail page indicators
    is_detail = (
        ("route.params" in content or "useRoute" in content)
        and not is_form
        and ("detail" in content_lower or "currentApproval" in content_lower
             or re.search(r"fetch\w+\(\s*id", content))
    )

    if is_edit:
        return "edit_page"
    if is_form:
        return "form_page"
    if is_list:
        return "list_page"
    if is_detail:
        return "detail_page"

    # Fallback by position (trained order)
    position_map = {
        0: "list_page",   # After types and composable (already parsed as non-vue)
        1: "form_page",
        2: "edit_page",
        3: "detail_page",
    }
    # Count how many vue files we've seen before this one
    return position_map.get(index - 2, f"page_{index}")


def _assign_path(file_type: str, entity_name: str | None, entity_lower: str | None, index: int) -> tuple[str, str]:
    """Assign file path and language based on detected file type."""
    e = entity_name or "Entity"
    el = entity_lower or "entity"

    paths = {
        "types": (f"types.ts", "typescript"),
        "composable": (f"composables/use{e}.ts", "typescript"),
        "list_page": (f"pages/{el}/index.vue", "vue"),
        "form_page": (f"pages/{el}/create.vue", "vue"),
        "edit_page": (f"pages/{el}/[id]/edit.vue", "vue"),
        "detail_page": (f"pages/{el}/[id]/index.vue", "vue"),
    }

    result = paths.get(file_type)
    if result:
        return result

    # Fallback
    if file_type.startswith("page_"):
        return (f"pages/{el}/page_{index}.vue", "vue")
    return (f"file_{index}.ts", "typescript")


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
