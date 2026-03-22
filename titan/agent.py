"""
Titan Agent — Main Orchestrator

Orchestrates full ERP module generation:
1. Select architecture → 2. Backend (multi-turn) → 3. Extract module_def →
4. Frontend (single-shot) → 5. Validate → 6. Debug (Groq) → 7. Package
"""

import asyncio
import inspect
import logging
import re
import time
import io
import zipfile
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from titan.config import settings
from titan.providers.runpod import RunPodProvider, RunPodResponse
from titan.providers.groq import GroqProvider
from titan.erp.prompts import (
    TITAN_BACKEND_SYSTEM_PROMPT,
    TITAN_UI_SYSTEM_PROMPT,
    MODULE_REGISTRY,
    MODULE_DESCRIPTIONS,
    FILE_ORDER,
    ARCH_LABELS,
    build_backend_turn1_prompt,
    build_backend_subsequent_prompt,
    build_frontend_prompt,
    build_frontend_fallback_prompt,
)
from titan.erp.parser import (
    parse_backend_response,
    parse_frontend_response,
    strip_think_blocks,
)
from titan.erp.validator import (
    validate_backend_module,
    validate_frontend_module,
)
from titan.erp.debugger import TitanDebugger

logger = logging.getLogger(__name__)

EventCallback = Optional[Callable[[str, Any], None]]


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class GeneratedFile:
    path: str
    content: str
    language: str = "python"


@dataclass
class ModuleResult:
    module_name: str
    architecture: str
    backend_files: list[GeneratedFile] = field(default_factory=list)
    frontend_files: list[GeneratedFile] = field(default_factory=list)
    backend_pass_rate: float = 0.0
    frontend_pass_rate: float = 0.0
    debug_iterations: int = 0
    generation_time_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        return self.backend_pass_rate >= 0.8 and self.frontend_pass_rate >= 0.8


@dataclass
class ERPBuildResult:
    modules: list[ModuleResult] = field(default_factory=list)
    scaffold_files: dict[str, str] = field(default_factory=dict)
    total_time_seconds: float = 0.0
    total_files: int = 0
    overall_pass_rate: float = 0.0

    def summary(self) -> str:
        lines = [
            f"ERP Build Complete: {len(self.modules)} module(s), "
            f"{self.total_files} files, {self.total_time_seconds:.1f}s",
            f"Overall pass rate: {self.overall_pass_rate:.0%}",
            "",
        ]
        for m in self.modules:
            status = "✅" if m.is_acceptable else "⚠️"
            lines.append(
                f"  {status} {m.module_name} ({m.architecture}): "
                f"backend={m.backend_pass_rate:.0%}, "
                f"frontend={m.frontend_pass_rate:.0%}, "
                f"debug_rounds={m.debug_iterations}"
            )
            if m.errors:
                for e in m.errors:
                    lines.append(f"       ⚠ {e}")
        return "\n".join(lines)


# =============================================================================
# Python type → TypeScript type mapping
# =============================================================================

_PY_TO_TS = {
    "str": "string", "String": "string",
    "int": "number", "Integer": "number",
    "float": "number", "Float": "number", "Decimal": "number",
    "bool": "boolean", "Boolean": "boolean",
    "date": "string", "datetime": "string", "Date": "string", "DateTime": "string",
}

_SKIP_FIELDS = {"id", "tenant_id", "created_at", "updated_at"}


# =============================================================================
# Agent
# =============================================================================

class TitanAgent:
    """
    Standalone ERP generation agent.

    Tier 1: Titan-Backend + Titan-UI via RunPod Serverless
    Tier 2: Groq (Llama 3.1 70B) for validation debugging
    """

    def __init__(self, on_event: EventCallback = None):
        self.runpod = RunPodProvider()
        self.groq = GroqProvider()
        self.debugger = TitanDebugger(self.groq, max_retries=settings.max_debug_retries)
        self.on_event = on_event

    async def _emit(self, event: str, data: Any = None):
        logger.info(f"[titan] {event}: {data}")
        if self.on_event:
            try:
                result = self.on_event(event, data or {})
                if inspect.isawaitable(result):
                    await result
            except Exception:
                pass

    # ─── Architecture Selection ───────────────────────────────────────────

    def select_architecture(self, module_name: str) -> str:
        return MODULE_REGISTRY.get(module_name, "service_layer")

    # ─── Backend Generation (multi-turn) ──────────────────────────────────

    async def generate_backend(self, module_name: str, arch: str) -> list[GeneratedFile]:
        file_order = FILE_ORDER.get(arch, [])
        if not file_order:
            raise ValueError(f"No file order for architecture: {arch}")

        await self._emit("backend.start", {"module": module_name, "arch": arch, "files": len(file_order)})

        conversation = []
        generated = []

        for i, file_path in enumerate(file_order):
            # Build prompt
            if i == 0:
                user_prompt = build_backend_turn1_prompt(module_name, arch)
            else:
                user_prompt = build_backend_subsequent_prompt(file_path)

            conversation.append({"role": "user", "content": user_prompt})

            await self._emit("backend.file", {
                "module": module_name, "file": file_path,
                "turn": i + 1, "total": len(file_order),
            })

            # Call Titan-Backend
            response = await self.runpod.chat(
                messages=conversation,
                model="titan-backend",
                system=TITAN_BACKEND_SYSTEM_PROMPT,
                temperature=settings.titan_temperature,
                max_tokens=settings.titan_max_tokens,
            )

            # Parse
            parsed = parse_backend_response(response.content)
            if parsed:
                generated.append(GeneratedFile(
                    path=f"{module_name}/{file_path}",
                    content=parsed[0].content,
                    language="python",
                ))
            else:
                generated.append(GeneratedFile(
                    path=f"{module_name}/{file_path}",
                    content=strip_think_blocks(response.content),
                    language="python",
                ))

            # Accumulate context
            conversation.append({"role": "assistant", "content": response.content})

            await self._emit("backend.file.done", {
                "file": file_path,
                "tokens_in": response.input_tokens,
                "tokens_out": response.output_tokens,
                "time_ms": response.execution_time_ms,
            })

        await self._emit("backend.done", {"module": module_name, "files": len(generated)})
        return generated

    # ─── module_def Extraction ────────────────────────────────────────────

    async def extract_module_def(self, module_name: str, backend_files: list[GeneratedFile]) -> Optional[dict]:
        """Parse generated backend code to build a module_def for the frontend prompt."""
        models_file = None
        router_file = None

        for f in backend_files:
            p = f.path.lower()
            if p.endswith("models.py") or p.endswith("orm_models.py") or p.endswith("write_models.py"):
                if models_file is None:
                    models_file = f
            if p.endswith("router.py"):
                router_file = f

        if not models_file:
            return None

        code = models_file.content

        # Extract entity class names
        class_pattern = re.compile(r"class\s+(\w+)\s*\(.*(?:Base|Model|DeclarativeBase)")
        class_matches = class_pattern.findall(code)
        entity_classes = [c for c in class_matches if not c.endswith("Mixin") and c != "Base"]

        if not entity_classes:
            return None

        primary_entity = entity_classes[0]
        entity_plural = self._pluralize(primary_entity)

        # Extract primary entity body
        body_pattern = re.compile(
            rf"class\s+{re.escape(primary_entity)}\s*\([^)]*\):(.*?)(?=\nclass\s|\Z)",
            re.DOTALL,
        )
        body_match = body_pattern.search(code)
        body = body_match.group(1) if body_match else code

        # Extract fields
        field_pattern = re.compile(r"(\w+)\s*:\s*Mapped\[(?:Optional\[)?(\w+)")
        fields = []
        for name, py_type in field_pattern.findall(body):
            if name in _SKIP_FIELDS:
                continue
            ts_type = _PY_TO_TS.get(py_type, "string")
            component = self._infer_component(name, ts_type)

            # Check optionality
            is_optional = False
            for line in body.split("\n"):
                if f"{name}:" in line and "Mapped" in line:
                    is_optional = "Optional" in line or "nullable=True" in line
                    break

            fields.append({
                "name": name, "ts": ts_type,
                "component": component, "required": not is_optional,
            })

        if len(fields) < 2:
            return None

        # API prefix from router
        api_prefix = f"/api/{module_name}"
        if router_file:
            prefix_match = re.search(r'prefix\s*=\s*["\']([^"\']+)["\']', router_file.content)
            if prefix_match:
                api_prefix = prefix_match.group(1)

        # Detect status enums
        statuses = None
        enum_match = re.search(r"class\s+(\w*(?:Status|State|Stage))\s*\(.*(?:str|Enum)", code)
        if enum_match:
            enum_name = enum_match.group(1)
            enum_body_pattern = re.compile(
                rf'class\s+{re.escape(enum_name)}\s*\([^)]*\):\s*\n((?:\s+\w+\s*=\s*["\'][^"\']+["\']\s*\n?)+)'
            )
            enum_body_match = enum_body_pattern.search(code)
            if enum_body_match:
                statuses = {}
                for val_match in re.finditer(r'(\w+)\s*=\s*["\']([^"\']+)["\']', enum_body_match.group(1)):
                    statuses[val_match.group(2)] = val_match.group(1).replace("_", " ").title()

        # Detect line items (master-detail)
        has_lines = False
        line_entity = None
        line_fields = None
        if len(entity_classes) > 1:
            child_class = entity_classes[1]
            child_pattern = re.compile(
                rf"class\s+{re.escape(child_class)}\s*\([^)]*\):(.*?)(?=\nclass\s|\Z)",
                re.DOTALL,
            )
            child_match = child_pattern.search(code)
            if child_match:
                child_body = child_match.group(1)
                if f"{primary_entity.lower()}_id" in child_body or "ForeignKey" in child_body:
                    has_lines = True
                    line_entity = child_class
                    line_fields = []
                    for fn, pt in field_pattern.findall(child_body):
                        if fn in _SKIP_FIELDS or fn.endswith("_id"):
                            continue
                        ts = _PY_TO_TS.get(pt, "string")
                        line_fields.append({
                            "name": fn, "ts": ts,
                            "component": self._infer_component(fn, ts),
                            "required": True,
                        })

        await self._emit("module_def.extracted", {
            "entity": primary_entity, "fields": len(fields),
            "has_lines": has_lines, "has_statuses": statuses is not None,
        })

        return {
            "api_prefix": api_prefix,
            "entity": primary_entity,
            "entity_plural": entity_plural,
            "fields": fields,
            "statuses": statuses,
            "has_lines": has_lines,
            "line_fields": line_fields,
            "line_entity": line_entity,
        }

    # ─── Frontend Generation (single-shot) ────────────────────────────────

    async def generate_frontend(
        self, module_name: str, module_def: Optional[dict] = None
    ) -> list[GeneratedFile]:
        await self._emit("frontend.start", {"module": module_name})

        if module_def:
            user_prompt = build_frontend_prompt(
                module_name=module_name,
                api_prefix=module_def.get("api_prefix", f"/api/{module_name}"),
                entity=module_def.get("entity", module_name.title()),
                entity_plural=module_def.get("entity_plural", f"{module_name.title()}s"),
                fields=module_def.get("fields", []),
                statuses=module_def.get("statuses"),
                has_lines=module_def.get("has_lines", False),
                line_fields=module_def.get("line_fields"),
                line_entity=module_def.get("line_entity"),
            )
        else:
            user_prompt = build_frontend_fallback_prompt(module_name)
            logger.warning(f"Using fallback frontend prompt for {module_name}")

        response = await self.runpod.chat(
            messages=[{"role": "user", "content": user_prompt}],
            model="titan-ui",
            system=TITAN_UI_SYSTEM_PROMPT,
            temperature=settings.titan_temperature,
            max_tokens=settings.titan_max_tokens,
        )

        # Log raw response length for debugging
        raw_len = len(response.content)
        separator_count = response.content.count("\n---\n")
        logger.info(
            f"Frontend raw response: {raw_len} chars, "
            f"{separator_count} '---' separators, "
            f"{response.output_tokens} output tokens"
        )

        parsed = parse_frontend_response(response.content)
        generated = []
        for pf in parsed:
            generated.append(GeneratedFile(
                path=f"frontend/modules/{module_name}/{pf.path}",
                content=pf.content,
                language=pf.language,
            ))
            logger.info(f"  Frontend file: {pf.path} ({len(pf.content)} chars)")

        await self._emit("frontend.done", {
            "module": module_name,
            "files": [g.path.split("/")[-1] for g in generated],
        })
        return generated

    # ─── Validate + Debug ─────────────────────────────────────────────────

    async def validate_and_debug(
        self,
        module_name: str,
        backend_files: list[GeneratedFile],
        frontend_files: list[GeneratedFile],
    ) -> tuple[list[GeneratedFile], list[GeneratedFile], float, float, int]:
        """Validate and fix. Returns (backend, frontend, be_rate, fe_rate, debug_count)."""

        # Validate backend
        be_dict = {f.path: f.content for f in backend_files}
        be_val = validate_backend_module(module_name, be_dict)

        # Validate frontend
        fe_dict = {f.path: f.content for f in frontend_files}
        fe_val = validate_frontend_module(module_name, fe_dict)

        await self._emit("validation", {
            "module": module_name,
            "backend_rate": be_val.pass_rate,
            "frontend_rate": fe_val.pass_rate,
            "backend_failures": len(be_val.all_failures),
            "frontend_failures": len(fe_val.all_failures),
        })

        debug_count = 0

        # Debug backend
        if be_val.all_failures:
            await self._emit("debug.backend.start", {"failures": len(be_val.all_failures)})
            be_dict = await self.debugger.debug_files(be_dict, be_val, "backend")
            be_val = validate_backend_module(module_name, be_dict)
            backend_files = [
                GeneratedFile(path=p, content=c, language="python")
                for p, c in be_dict.items()
            ]
            debug_count += 1
            await self._emit("debug.backend.done", {"pass_rate": be_val.pass_rate})

        # Debug frontend
        if fe_val.all_failures:
            await self._emit("debug.frontend.start", {"failures": len(fe_val.all_failures)})
            fe_dict = await self.debugger.debug_files(fe_dict, fe_val, "frontend")
            fe_val = validate_frontend_module(module_name, fe_dict)
            frontend_files = [
                GeneratedFile(path=p, content=c, language=pf_lang)
                for p, c in fe_dict.items()
                for pf_lang in [("vue" if p.endswith(".vue") else "typescript")]
            ]
            debug_count += 1
            await self._emit("debug.frontend.done", {"pass_rate": fe_val.pass_rate})

        return backend_files, frontend_files, be_val.pass_rate, fe_val.pass_rate, debug_count

    # ─── Full Module Build ────────────────────────────────────────────────

    async def build_module(self, module_name: str) -> ModuleResult:
        start = time.time()
        arch = self.select_architecture(module_name)
        result = ModuleResult(module_name=module_name, architecture=arch)

        await self._emit("module.start", {"module": module_name, "arch": arch})

        try:
            # 1. Backend
            result.backend_files = await self.generate_backend(module_name, arch)

            # 2. Extract module_def
            module_def = await self.extract_module_def(module_name, result.backend_files)

            # 3. Frontend
            result.frontend_files = await self.generate_frontend(module_name, module_def)

            # 4. Validate + Debug
            (
                result.backend_files,
                result.frontend_files,
                result.backend_pass_rate,
                result.frontend_pass_rate,
                result.debug_iterations,
            ) = await self.validate_and_debug(
                module_name, result.backend_files, result.frontend_files
            )

        except Exception as e:
            result.errors.append(str(e))
            logger.exception(f"Failed to build {module_name}: {e}")
            await self._emit("module.error", {"module": module_name, "error": str(e)})

        result.generation_time_seconds = time.time() - start
        await self._emit("module.done", {
            "module": module_name,
            "time": f"{result.generation_time_seconds:.1f}s",
            "backend_rate": f"{result.backend_pass_rate:.0%}",
            "frontend_rate": f"{result.frontend_pass_rate:.0%}",
        })
        return result

    # ─── Full ERP Build ───────────────────────────────────────────────────

    async def build_erp(self, module_names: list[str]) -> ERPBuildResult:
        start = time.time()
        build = ERPBuildResult()

        await self._emit("erp.start", {"modules": module_names, "count": len(module_names)})

        for name in module_names:
            result = await self.build_module(name)
            build.modules.append(result)

        # Calculate totals
        build.total_time_seconds = time.time() - start
        build.total_files = sum(
            len(m.backend_files) + len(m.frontend_files)
            for m in build.modules
        )

        rates = []
        for m in build.modules:
            if m.backend_pass_rate > 0:
                rates.append(m.backend_pass_rate)
            if m.frontend_pass_rate > 0:
                rates.append(m.frontend_pass_rate)
        build.overall_pass_rate = sum(rates) / len(rates) if rates else 0.0

        await self._emit("erp.done", {
            "modules": len(build.modules),
            "files": build.total_files,
            "time": f"{build.total_time_seconds:.1f}s",
            "pass_rate": f"{build.overall_pass_rate:.0%}",
        })

        return build

    # ─── Packaging ────────────────────────────────────────────────────────

    def package_zip(self, result: ERPBuildResult) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for module in result.modules:
                for f in module.backend_files:
                    zf.writestr(f"backend/modules/{f.path}", f.content)
                for f in module.frontend_files:
                    zf.writestr(f.path, f.content)
            zf.writestr("BUILD_SUMMARY.txt", result.summary())
        buf.seek(0)
        return buf.read()

    # ─── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _pluralize(name: str) -> str:
        if name.endswith(("s", "x", "sh", "ch")):
            return name + "es"
        if name.endswith("y") and name[-2:] not in ("ay", "ey", "oy", "uy"):
            return name[:-1] + "ies"
        return name + "s"

    @staticmethod
    def _infer_component(field_name: str, ts_type: str) -> str:
        n = field_name.lower()
        if ts_type == "boolean":
            return "Checkbox"
        if any(k in n for k in ("date", "_at", "_on")):
            return "DatePicker"
        if any(k in n for k in ("amount", "price", "total", "cost", "revenue", "salary", "fee", "balance")):
            return "InputNumber"
        if any(k in n for k in ("description", "notes", "comment", "body", "content", "address")):
            return "Textarea"
        if any(k in n for k in ("status", "state", "type", "priority", "category", "stage", "level")):
            return "Select"
        if ts_type == "number":
            return "InputNumber"
        return "InputText"

    async def close(self):
        await self.runpod.close()
        await self.groq.close()
