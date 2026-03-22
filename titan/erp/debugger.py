"""
Groq-powered Tier-2 Debugger — Per-File Strategy

Three key improvements over the initial version:
1. Debug ONE file at a time (not all at once) — Groq handles focused tasks better
2. Detailed logging of what Groq returns and what fixes are applied
3. Robust path matching for fix application
"""

import logging
from typing import Optional

from titan.providers.groq import GroqProvider
from titan.erp.parser import parse_debug_response, ParsedFile
from titan.erp.validator import ModuleValidation, validate_backend_file, validate_frontend_file

logger = logging.getLogger(__name__)


# =============================================================================
# Architecture-aware fix descriptions
# =============================================================================

BACKEND_FIX_DESCRIPTIONS = {
    "sqlalchemy_2_0_style": {
        "rule": "Use SQLAlchemy 2.0 Mapped[] + mapped_column() syntax. Never use legacy Column().",
        "fix": (
            "Replace:\n"
            "  name = Column(String(100), nullable=False)\n"
            "With:\n"
            "  name: Mapped[str] = mapped_column(String(100), nullable=False)\n"
            "Also add import: from sqlalchemy.orm import Mapped, mapped_column"
        ),
    },
    "tenant_id_present": {
        "rule": "Every ORM model MUST include tenant_id for multi-tenancy.",
        "fix": (
            "Add this field to every ORM model class:\n"
            "  tenant_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)"
        ),
    },
    "cascade_delete_orphan": {
        "rule": "Parent-child ORM relationships MUST use cascade='all, delete-orphan' "
                "so deleting a parent automatically removes orphaned children.",
        "fix": (
            'Add cascade to every parent-side relationship:\n'
            '  items: Mapped[list["OrderLine"]] = relationship(\n'
            '      "OrderLine",\n'
            '      back_populates="order",\n'
            '      cascade="all, delete-orphan",\n'
            '      lazy="selectin",\n'
            '  )'
        ),
    },
    "timestamps_func_now": {
        "rule": "Timestamps must use server_default=func.now() for created_at and "
                "onupdate=func.now() for updated_at. Never use datetime.utcnow.",
        "fix": (
            "Replace:\n"
            "  created_at = Column(DateTime, default=datetime.utcnow)\n"
            "With:\n"
            "  created_at: Mapped[datetime] = mapped_column(\n"
            "      DateTime(timezone=True), server_default=func.now()\n"
            "  )\n"
            "  updated_at: Mapped[Optional[datetime]] = mapped_column(\n"
            "      DateTime(timezone=True), onupdate=func.now()\n"
            "  )\n"
            "Import: from sqlalchemy import func"
        ),
    },
    "monetary_numeric_not_float": {
        "rule": "Monetary fields (price, amount, total, cost, fee) MUST use Numeric(12,2), never Float. "
                "Float causes rounding errors in financial calculations.",
        "fix": (
            "Replace:\n"
            "  price: Mapped[float] = mapped_column(Float)\n"
            "With:\n"
            "  price: Mapped[Decimal] = mapped_column(Numeric(12, 2))\n"
            "Import: from decimal import Decimal; from sqlalchemy import Numeric"
        ),
    },
    "bilingual_fields": {
        "rule": "User-facing entity names must include bilingual fields: name_en (English) and name_ar (Arabic).",
        "fix": (
            "If name_en exists, also add:\n"
            "  name_ar: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)"
        ),
    },
    "schema_from_attributes": {
        "rule": "All Pydantic Response/Read schemas MUST include model_config = ConfigDict(from_attributes=True) "
                "for SQLAlchemy ORM compatibility.",
        "fix": (
            "Add to every Response/Read schema class:\n"
            "  model_config = ConfigDict(from_attributes=True)\n"
            "Import: from pydantic import ConfigDict"
        ),
    },
    "async_endpoints": {
        "rule": "All FastAPI route handlers MUST be async def, not def, for non-blocking database access.",
        "fix": (
            "Replace:\n"
            "  @router.get('/')\n"
            "  def list_items(...):\n"
            "With:\n"
            "  @router.get('/')\n"
            "  async def list_items(...):"
        ),
    },
    "no_sql_injection": {
        "rule": "NEVER use f-strings or string formatting in SQL queries. Always use parameterized queries.",
        "fix": (
            "Replace:\n"
            '  query = f"SELECT * FROM items WHERE name = \'{name}\'"\n'
            "With:\n"
            "  stmt = select(Item).where(Item.name == name)"
        ),
    },
}

FRONTEND_FIX_DESCRIPTIONS = {
    "typescript_script_setup": {
        "rule": "Every .vue file must use <script setup lang=\"ts\">.",
        "fix": 'Replace <script setup> with <script setup lang="ts">',
    },
    "primevue_components": {
        "rule": "Use PrimeVue components exclusively. Never raw HTML <input>, <table>, or <select>.",
        "fix": "Replace <input> with <InputText>, <table> with <DataTable>, <select> with <Select>.",
    },
    "no_inline_styles": {
        "rule": "Use Tailwind CSS classes only. Never inline styles or <style> blocks.",
        "fix": 'Replace style="..." with Tailwind classes like class="p-4 bg-white rounded-lg"',
    },
    "uses_dollar_fetch": {
        "rule": "Use Nuxt 3 $fetch for API calls. Never axios or hardcoded URLs.",
        "fix": "Use $fetch with useRuntimeConfig().public.apiBase for all API calls.",
    },
    "bearer_auth": {
        "rule": "Include Authorization: Bearer token on every API call.",
        "fix": "Add headers: { Authorization: `Bearer ${token}` } to every $fetch call.",
    },
    "tenant_id_in_api": {
        "rule": "Include tenant_id in API call parameters or headers.",
        "fix": "Add tenant_id from auth store: params: { tenant_id: auth.tenantId }",
    },
    "loading_state": {
        "rule": "Every data page must have a loading state using ref(false).",
        "fix": "Add: const loading = ref(false) and toggle it around API calls.",
    },
    "skeleton_loader": {
        "rule": "List/index pages must show PrimeVue Skeleton loaders while loading.",
        "fix": 'Add: <Skeleton v-if="loading" class="mb-2" height="2rem" /> before the DataTable.',
    },
    "empty_state": {
        "rule": "List pages must show an empty state message when no data exists.",
        "fix": 'Add: <div v-if="!loading && items.length === 0">No records found</div>',
    },
    "zod_validation": {
        "rule": "Form/create pages must use Zod schemas for client-side validation.",
        "fix": "Add: import { z } from 'zod'; const schema = z.object({ name_en: z.string().min(1) })",
    },
    "define_page_meta": {
        "rule": "Every page must call definePageMeta({ middleware: ['auth'] }).",
        "fix": "Add at top of <script setup>: definePageMeta({ middleware: ['auth'] })",
    },
    "bilingual_fields": {
        "rule": "Forms with name_en must also include name_ar with dir=\"rtl\".",
        "fix": 'Add: <InputText v-model="form.name_ar" dir="rtl" placeholder="الاسم بالعربي" />',
    },
    "try_catch_error_handling": {
        "rule": "Every API call must be wrapped in try/catch/finally.",
        "fix": "Wrap $fetch calls in try/catch/finally with loading state toggling.",
    },
    "toast_notifications": {
        "rule": "Show PrimeVue Toast on success/error for every API mutation.",
        "fix": "Add: const toast = useToast(); toast.add({severity:'success', ...}) after API calls.",
    },
    "no_think_tags": {
        "rule": "Model <think> reasoning blocks must not appear in final code.",
        "fix": "Remove all <think>...</think> blocks from the output.",
    },
}


# =============================================================================
# Debug system prompt
# =============================================================================

DEBUGGER_SYSTEM_PROMPT = """You are a code debugger for Crystal Helix ERP.
You fix code that was generated by a fine-tuned model.

CRITICAL RULES:
1. Apply ONLY the specific fixes described — do not rewrite or restructure anything
2. Keep ALL existing code, imports, class names, function names exactly as they are
3. Only add/modify the specific things mentioned in the fix instructions
4. Output the COMPLETE fixed file (not just the changed parts)

Output format:
### FILE: <exact_same_path_as_given>
```python
<complete fixed file>
```"""


# =============================================================================
# Debugger class
# =============================================================================

class TitanDebugger:
    """
    Per-file architecture-aware Tier-2 debugger using Groq.

    Instead of sending all failures in one batch, we debug one file at a time.
    This gives Groq a focused task and produces more accurate fixes.
    """

    def __init__(self, groq: GroqProvider, max_retries: int = 3):
        self.groq = groq
        self.max_retries = max_retries

    async def debug_files(
        self,
        files: dict[str, str],
        validation: ModuleValidation,
        file_type: str = "backend",
    ) -> dict[str, str]:
        """
        Fix validation failures using Groq, one file at a time.

        Returns: Updated files dict with fixes applied.
        """
        if not validation.all_failures:
            logger.info(f"No {file_type} failures to debug")
            return files

        fix_db = BACKEND_FIX_DESCRIPTIONS if file_type == "backend" else FRONTEND_FIX_DESCRIPTIONS

        # Group failures by file
        failures_by_file: dict[str, list[str]] = {}
        for file_path, check_name in validation.all_failures:
            failures_by_file.setdefault(file_path, []).append(check_name)

        logger.info(
            f"Debug {file_type}: {len(validation.all_failures)} failures "
            f"across {len(failures_by_file)} file(s)"
        )

        # Debug each file independently
        for file_path, check_names in failures_by_file.items():
            code = files.get(file_path, "")
            if not code:
                # Try matching by suffix
                for path, content in files.items():
                    if path.endswith(file_path.split("/")[-1]):
                        code = content
                        file_path = path  # Use the actual path
                        break

            if not code:
                logger.warning(f"Could not find code for {file_path}, skipping")
                continue

            logger.info(f"Debugging {file_path}: {check_names}")

            # Retry loop for this single file
            for attempt in range(self.max_retries):
                prompt = self._build_single_file_prompt(
                    file_path, code, check_names, fix_db, file_type
                )

                try:
                    response = await self.groq.chat(
                        messages=[{"role": "user", "content": prompt}],
                        system=DEBUGGER_SYSTEM_PROMPT,
                    )

                    # Log first 200 chars of response for debugging
                    logger.info(
                        f"Groq response for {file_path} (attempt {attempt+1}): "
                        f"{response[:200]}..."
                    )

                    fixes = parse_debug_response(response)

                    if not fixes:
                        logger.warning(
                            f"No parseable fixes from Groq for {file_path} "
                            f"(attempt {attempt+1})"
                        )
                        continue

                    # Apply the fix (take the first matching fix)
                    applied = False
                    for fix in fixes:
                        # Accept the fix if it broadly matches this file
                        file_basename = file_path.split("/")[-1]
                        fix_basename = fix.path.split("/")[-1]

                        if (fix_basename == file_basename or
                            fix.path in file_path or
                            file_path.endswith(fix.path)):

                            old_len = len(code)
                            code = fix.content
                            files[file_path] = code
                            applied = True
                            logger.info(
                                f"Applied fix to {file_path}: "
                                f"{old_len} → {len(code)} chars"
                            )
                            break

                    if not applied:
                        logger.warning(
                            f"Groq returned fix for '{fixes[0].path}' but expected "
                            f"'{file_path}' — path mismatch"
                        )
                        # Force-apply if only one fix returned
                        if len(fixes) == 1:
                            code = fixes[0].content
                            files[file_path] = code
                            logger.info(f"Force-applied single fix to {file_path}")
                            applied = True

                    if not applied:
                        continue

                    # Re-validate just this file
                    if file_type == "backend":
                        result = validate_backend_file(file_path, code)
                    else:
                        result = validate_frontend_file(file_path, code)

                    remaining = result.failed
                    if not remaining:
                        logger.info(f"✅ {file_path} now passes all checks!")
                        break
                    else:
                        check_names = remaining
                        logger.info(
                            f"⚠ {file_path} still has {len(remaining)} failures "
                            f"after attempt {attempt+1}: {remaining}"
                        )

                except Exception as e:
                    logger.error(
                        f"Debug attempt {attempt+1} failed for {file_path}: {e}",
                        exc_info=True,
                    )

        return files

    def _build_single_file_prompt(
        self,
        file_path: str,
        code: str,
        check_names: list[str],
        fix_db: dict,
        file_type: str,
    ) -> str:
        """Build a focused debug prompt for a single file."""
        lang = "python" if file_type == "backend" else "vue"

        parts = [
            f"Fix the following validation failures in this {file_type} file.",
            f"File path: {file_path}",
            "",
        ]

        for check_name in check_names:
            fix_info = fix_db.get(check_name, {})
            rule = fix_info.get("rule", f"Check '{check_name}' failed.")
            fix = fix_info.get("fix", "Apply the appropriate fix.")

            parts.append(f"❌ FAILED: {check_name}")
            parts.append(f"   Rule: {rule}")
            parts.append(f"   Fix:  {fix}")
            parts.append("")

        parts.append(f"Current code:")
        parts.append(f"### FILE: {file_path}")
        parts.append(f"```{lang}")
        parts.append(code)
        parts.append("```")
        parts.append("")
        parts.append(
            f"Output the COMPLETE corrected file using the EXACT same path "
            f"(### FILE: {file_path})."
        )

        return "\n".join(parts)
