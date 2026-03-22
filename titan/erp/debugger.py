"""
Groq-powered Tier-2 Debugger

The KEY improvement over Cortex: architecture-aware debug prompts.
Instead of just saying "FAILED: cascade_delete_orphan", we tell the LLM
exactly what the rule means and show a concrete code fix.

This dramatically increases fix accuracy because the LLM doesn't have to
guess what "cascade_delete_orphan" means.
"""

import logging
from typing import Optional

from titan.providers.groq import GroqProvider
from titan.erp.parser import parse_debug_response, ParsedFile
from titan.erp.validator import ModuleValidation

logger = logging.getLogger(__name__)


# =============================================================================
# Architecture-aware fix descriptions
# =============================================================================

# For each validation check, we provide:
# 1. What the check means (in plain English)
# 2. Why it matters (architectural rule from the Titan system prompt)
# 3. A concrete before/after code example

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
        "fix": (
            "Replace <input> with <InputText>, <table> with <DataTable>, <select> with <Select>.\n"
            "Example: <InputText v-model=\"form.name_en\" />"
        ),
    },
    "no_inline_styles": {
        "rule": "Use Tailwind CSS classes only. Never inline styles or <style> blocks.",
        "fix": 'Replace style="..." with Tailwind classes like class="p-4 bg-white rounded-lg"',
    },
    "uses_dollar_fetch": {
        "rule": "Use Nuxt 3 $fetch for API calls. Never axios or hardcoded URLs.",
        "fix": (
            "Replace axios.get(...) with $fetch(...) using useRuntimeConfig():\n"
            "  const config = useRuntimeConfig()\n"
            "  const data = await $fetch(`${config.public.apiBase}/api/items`)"
        ),
    },
    "bearer_auth": {
        "rule": "Include Authorization: Bearer token on every API call.",
        "fix": (
            "Add headers to $fetch:\n"
            "  const { token } = useAuth()\n"
            "  await $fetch(url, { headers: { Authorization: `Bearer ${token}` } })"
        ),
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
        "fix": "Add: <Skeleton v-if=\"loading\" class=\"mb-2\" height=\"2rem\" /> before the DataTable.",
    },
    "empty_state": {
        "rule": "List pages must show an empty state message when no data exists.",
        "fix": 'Add: <div v-if="!loading && items.length === 0">No records found</div>',
    },
    "zod_validation": {
        "rule": "Form/create pages must use Zod schemas for client-side validation.",
        "fix": (
            "Add Zod schema:\n"
            "  import { z } from 'zod'\n"
            "  const schema = z.object({ name_en: z.string().min(1) })"
        ),
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
        "fix": (
            "Wrap $fetch calls:\n"
            "  try { loading.value = true; data = await $fetch(...) }\n"
            "  catch(e) { toast.add({severity:'error',...}) }\n"
            "  finally { loading.value = false }"
        ),
    },
    "toast_notifications": {
        "rule": "Show PrimeVue Toast on success/error for every API mutation.",
        "fix": (
            "Add: const toast = useToast()\n"
            "On success: toast.add({severity:'success', summary:'Saved', life:3000})\n"
            "On error: toast.add({severity:'error', summary:'Error', detail: e.message, life:5000})"
        ),
    },
    "no_think_tags": {
        "rule": "Model <think> reasoning blocks must not appear in final code.",
        "fix": "Remove all <think>...</think> blocks from the output.",
    },
}


# =============================================================================
# Debug system prompt (architecture-aware)
# =============================================================================

DEBUGGER_SYSTEM_PROMPT = """You are a code debugger for Crystal Helix ERP.
You fix code that was generated by a fine-tuned model.

Your job is to apply ONLY the specific fixes described below.
Do NOT rewrite the code, do NOT add new features, do NOT change the structure.
Just fix the exact issues listed.

For each fixed file, output it with:
### FILE: <path>
```<language>
<fixed code>
```

Keep the code structure, naming, and logic exactly the same — only fix the listed issues."""


# =============================================================================
# Debugger class
# =============================================================================

class TitanDebugger:
    """
    Architecture-aware Tier-2 debugger using Groq.

    Instead of sending generic "fix this" prompts, we send:
    1. The exact architectural rule being violated
    2. A concrete before/after code example
    3. The failing file's full content

    This gives the Groq model enough context to make precise, targeted fixes.
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
        Fix validation failures using Groq.

        Args:
            files: dict of {path: content} for all generated files
            validation: validation result with failures
            file_type: "backend" or "frontend"

        Returns:
            Updated files dict with fixes applied
        """
        if not validation.all_failures:
            return files

        fix_db = BACKEND_FIX_DESCRIPTIONS if file_type == "backend" else FRONTEND_FIX_DESCRIPTIONS

        for attempt in range(self.max_retries):
            if not validation.all_failures:
                break

            logger.info(
                f"Debug attempt {attempt + 1}/{self.max_retries}: "
                f"{len(validation.all_failures)} failures in {file_type}"
            )

            prompt = self._build_prompt(files, validation, fix_db, file_type)

            try:
                response = await self.groq.chat(
                    messages=[{"role": "user", "content": prompt}],
                    system=DEBUGGER_SYSTEM_PROMPT,
                )

                fixes = parse_debug_response(response)
                if not fixes:
                    logger.warning("Groq returned no parseable fixes")
                    continue

                # Apply fixes
                applied = 0
                for fix in fixes:
                    for path in files:
                        if fix.path in path or path.endswith(fix.path):
                            files[path] = fix.content
                            applied += 1
                            break

                logger.info(f"Applied {applied} fix(es)")

                # Re-validate
                if file_type == "backend":
                    from titan.erp.validator import validate_backend_module
                    validation = validate_backend_module(validation.module_name, files)
                else:
                    from titan.erp.validator import validate_frontend_module
                    validation = validate_frontend_module(validation.module_name, files)

                if not validation.all_failures:
                    logger.info(f"All {file_type} checks now pass! ✅")
                    break

            except Exception as e:
                logger.error(f"Debug attempt {attempt + 1} failed: {e}")

        return files

    def _build_prompt(
        self,
        files: dict[str, str],
        validation: ModuleValidation,
        fix_db: dict,
        file_type: str,
    ) -> str:
        """Build an architecture-aware debug prompt."""
        parts = [
            f"The following {file_type} files have validation failures. "
            f"Fix ONLY the issues listed below. Preserve ALL other code exactly.",
            "",
        ]

        # Group failures by file
        failures_by_file: dict[str, list[str]] = {}
        for file_path, check_name in validation.all_failures:
            failures_by_file.setdefault(file_path, []).append(check_name)

        for file_path, check_names in failures_by_file.items():
            parts.append(f"═══ File: {file_path} ═══")
            parts.append("")

            for check_name in check_names:
                fix_info = fix_db.get(check_name, {})
                rule = fix_info.get("rule", f"Check '{check_name}' failed.")
                fix = fix_info.get("fix", "Apply the appropriate fix.")

                parts.append(f"❌ FAILED: {check_name}")
                parts.append(f"   Rule: {rule}")
                parts.append(f"   Fix : {fix}")
                parts.append("")

            # Include the current file content
            code = files.get(file_path, "")
            if not code:
                # Try matching by suffix
                for path, content in files.items():
                    if path.endswith(file_path) or file_path.endswith(path.split("/")[-1]):
                        code = content
                        break

            if code:
                lang = "python" if file_type == "backend" else "vue"
                parts.append(f"Current code:")
                parts.append(f"```{lang}")
                parts.append(code)
                parts.append("```")
                parts.append("")

        parts.append(
            "Output ONLY the corrected file(s). Use ### FILE: <path> followed by "
            "a fenced code block for each file."
        )

        return "\n".join(parts)
