"""
Validators for Titan-generated ERP code.

Backend: 9 regex-based checks from the training data system prompt.
Frontend: 15 regex-based checks from the validation scripts.
"""

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FileResult:
    file_path: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        total = len(self.passed) + len(self.failed)
        return len(self.passed) / total if total > 0 else 1.0

    @property
    def is_acceptable(self) -> bool:
        return self.pass_rate >= 0.8


@dataclass
class ModuleValidation:
    module_name: str
    file_results: list[FileResult] = field(default_factory=list)

    @property
    def total_checks(self) -> int:
        return sum(len(r.passed) + len(r.failed) for r in self.file_results)

    @property
    def total_passed(self) -> int:
        return sum(len(r.passed) for r in self.file_results)

    @property
    def pass_rate(self) -> float:
        return self.total_passed / self.total_checks if self.total_checks > 0 else 1.0

    @property
    def failed_files(self) -> list[FileResult]:
        return [r for r in self.file_results if not r.is_acceptable]

    @property
    def all_failures(self) -> list[tuple[str, str]]:
        """List of (file_path, check_name) for all failures."""
        result = []
        for fr in self.file_results:
            for check_name in fr.failed:
                result.append((fr.file_path, check_name))
        return result

    def summary(self) -> str:
        lines = [f"Module: {self.module_name} ({self.pass_rate:.0%} pass rate)"]
        for r in self.file_results:
            status = "✅" if r.is_acceptable else "❌"
            lines.append(f"  {status} {r.file_path}: {r.pass_rate:.0%}")
            for f in r.failed:
                lines.append(f"       ✗ {f}")
        return "\n".join(lines)


def _check(result: FileResult, name: str, condition: bool):
    if condition:
        result.passed.append(name)
    else:
        result.failed.append(name)


# ─── Backend checks ──────────────────────────────────────────────────────────


def validate_backend_file(file_path: str, code: str) -> FileResult:
    result = FileResult(file_path=file_path)

    # 1. SQLAlchemy 2.0 style
    if "models" in file_path or "orm" in file_path or "write_models" in file_path:
        has_mapped = bool(re.search(r"Mapped\[", code))
        has_legacy = bool(re.search(r"Column\(", code))
        _check(result, "sqlalchemy_2_0_style", has_mapped and not has_legacy)

        # 2. tenant_id
        _check(result, "tenant_id_present", "tenant_id" in code)

        # 3. cascade on relationships
        if "relationship" in code:
            has_cascade = bool(re.search(r'cascade\s*=\s*["\']all,\s*delete-orphan["\']', code))
            _check(result, "cascade_delete_orphan", has_cascade)

        # 4. Timestamps use func.now()
        if "created_at" in code or "updated_at" in code:
            uses_func_now = bool(re.search(r"func\.now\(\)", code))
            no_utcnow = "datetime.utcnow" not in code
            _check(result, "timestamps_func_now", uses_func_now and no_utcnow)

        # 5. Monetary fields use Numeric, not Float
        money_keywords = ("price", "amount", "total", "cost", "fee", "salary", "revenue", "balance")
        has_money_field = any(kw in code.lower() for kw in money_keywords)
        if has_money_field or "Numeric" in code or "Float" in code:
            uses_numeric = "Numeric" in code or "Float" not in code
            _check(result, "monetary_numeric_not_float", uses_numeric)

        # 6. Bilingual fields
        if "name_en" in code:
            _check(result, "bilingual_fields", "name_ar" in code)

    # 7. Schemas: ConfigDict(from_attributes=True)
    if "schemas" in file_path:
        if "Response" in code or "Read" in code:
            has_config = bool(re.search(r"from_attributes\s*=\s*True", code))
            _check(result, "schema_from_attributes", has_config)

    # 8. Router: async def + Depends
    if "router" in file_path:
        has_async = bool(re.search(r"async\s+def", code))
        _check(result, "async_endpoints", has_async)

    # 9. No SQL string interpolation
    has_f_sql = bool(re.search(r'f["\'].*(?:SELECT|INSERT|UPDATE|DELETE)', code, re.IGNORECASE))
    _check(result, "no_sql_injection", not has_f_sql)

    return result


# ─── Frontend checks ─────────────────────────────────────────────────────────


def validate_frontend_file(file_path: str, code: str) -> FileResult:
    result = FileResult(file_path=file_path)

    is_vue = file_path.endswith(".vue")
    is_ts = file_path.endswith(".ts")

    # 1. TypeScript setup
    if is_vue:
        has_ts = bool(re.search(r'<script\s+setup\s+lang=["\']ts["\']', code))
        _check(result, "typescript_script_setup", has_ts)

    # 2. PrimeVue components
    if is_vue:
        no_raw = (
            not bool(re.search(r"<input\s", code))
            and not bool(re.search(r"<table\s", code))
            and not bool(re.search(r"<select\s", code))
        )
        _check(result, "primevue_components", no_raw)

    # 3. No inline styles
    if is_vue:
        no_style_attr = not bool(re.search(r'style\s*=\s*["\']', code))
        no_style_block = not bool(re.search(r"<style", code))
        _check(result, "no_inline_styles", no_style_attr and no_style_block)

    # 4. $fetch usage
    if is_ts or is_vue:
        if "fetch" in code.lower() or "api" in code.lower():
            uses_fetch = "$fetch" in code or "useFetch" in code
            no_axios = "axios" not in code
            _check(result, "uses_dollar_fetch", uses_fetch and no_axios)

    # 5. Bearer auth
    if "$fetch" in code or "useFetch" in code:
        _check(result, "bearer_auth", "Bearer" in code)

    # 6. tenant_id in API calls
    if "$fetch" in code:
        _check(result, "tenant_id_in_api", "tenant_id" in code)

    # 7. Loading state
    if is_vue:
        has_loading = bool(re.search(r"ref\s*\(\s*(?:false|true)\s*\)", code)) or "loading" in code
        _check(result, "loading_state", has_loading)

    # 8. Skeleton loader
    if is_vue and ("list" in file_path.lower() or "index" in file_path.lower()):
        _check(result, "skeleton_loader", "Skeleton" in code)

    # 9. Empty state
    if is_vue and ("list" in file_path.lower() or "index" in file_path.lower()):
        has_empty = bool(re.search(r"empty|no\s+data|no\s+records", code, re.IGNORECASE))
        _check(result, "empty_state", has_empty)

    # 10. Zod validation on forms
    if is_vue and ("form" in file_path.lower() or "create" in file_path.lower()):
        has_zod = "z." in code or "schema" in code.lower()
        _check(result, "zod_validation", has_zod)

    # 11. definePageMeta with auth middleware
    if is_vue:
        _check(result, "define_page_meta", "definePageMeta" in code)

    # 12. Bilingual fields
    if is_vue and ("form" in file_path.lower() or "create" in file_path.lower()):
        if "name_en" in code:
            _check(result, "bilingual_fields", "name_ar" in code)

    # 13. try/catch on API calls
    if "$fetch" in code:
        _check(result, "try_catch_error_handling", "try" in code and "catch" in code)

    # 14. Toast notifications
    if "$fetch" in code:
        has_toast = bool(re.search(r"useToast|toast\.|Toast", code))
        _check(result, "toast_notifications", has_toast)

    # 15. No <think> tags
    _check(result, "no_think_tags", "<think>" not in code)

    return result


# ─── Module-level validation ─────────────────────────────────────────────────


def validate_backend_module(module_name: str, files: dict[str, str]) -> ModuleValidation:
    result = ModuleValidation(module_name=module_name)
    for path, code in files.items():
        result.file_results.append(validate_backend_file(path, code))
    return result


def validate_frontend_module(module_name: str, files: dict[str, str]) -> ModuleValidation:
    result = ModuleValidation(module_name=module_name)
    for path, code in files.items():
        result.file_results.append(validate_frontend_file(path, code))
    return result
