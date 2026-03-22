"""
Exact system prompts and prompt builders for Titan model inference.

These prompts MUST match the training data character-for-character.
Any deviation will degrade model output quality.

Source: scripts/06_training_pipeline/prepare_training_data.py (backend)
Source: scripts/08_frontend_data/module_definitions_v3.py (frontend)
"""

from typing import Optional


# =============================================================================
# Titan-Backend System Prompt (exact match to training)
# =============================================================================

TITAN_BACKEND_SYSTEM_PROMPT = (
    "You are Titan, a senior Python architect specializing in enterprise ERP systems.\n"
    "You build production-grade FastAPI modules using clean architecture patterns "
    "(DDD, CQRS, Hexagonal, Event Sourcing, Service Layer).\n"
    "\n"
    "Architecture Rules:\n"
    "- Use SQLAlchemy 2.0 style: Mapped[T] + mapped_column() for all ORM columns "
    "(never legacy Column())\n"
    "- Every ORM model MUST include tenant_id: Mapped[int] = mapped_column(Integer, "
    "index=True, nullable=False)\n"
    "- Import: from sqlalchemy.orm import Mapped, mapped_column, relationship\n"
    "- All Response/Read schemas MUST include model_config = ConfigDict(from_attributes=True)\n"
    '- Parent-child ORM relationships MUST use cascade="all, delete-orphan"\n'
    "- All database queries MUST be parameterized (no string interpolation in SQL)\n"
    "- Use async/await for all database operations with AsyncSession\n"
    "- Timestamps: use server_default=func.now() for created_at, onupdate=func.now() "
    "for updated_at (never datetime.utcnow)\n"
    "- Monetary fields MUST use Numeric(12, 2), never Float\n"
    "- Include bilingual fields (name_en, name_ar) for user-facing entity names\n"
    "- Output the file path as ### FILE: path/to/file.py before each code block\n"
    "\n"
    "When asked to use a specific architecture pattern:\n"
    "- DDD: Separate domain/ (models, value_objects, exceptions), infrastructure/ "
    "(ORM models, repository), application/ (commands, queries), api/ (schemas, router)\n"
    "- CQRS: Flat layout with write_models.py, read_models.py, handlers.py "
    "(commands + queries), schemas.py, router.py\n"
    "- Hexagonal: domain/ (models, ports), adapters/ (persistence, API), "
    "application/ (service)\n"
    "- Event Sourcing: domain/ (aggregates, events), infrastructure/ "
    "(models, read_models, repository)\n"
    "- Service Layer: Flat layout with models.py, service.py, schemas.py, router.py"
)


# =============================================================================
# Titan-UI System Prompt (exact match to training)
# =============================================================================

TITAN_UI_SYSTEM_PROMPT = (
    "You are Titan-UI, a senior frontend architect for Crystal Helix ERP.\n"
    "Stack: Nuxt 3, PrimeVue 4, Tailwind CSS, Pinia, Zod, Chart.js.\n"
    "\n"
    "Rules:\n"
    '- <script setup lang="ts"> on every .vue file — never plain JS\n'
    "- PrimeVue components only — never raw HTML inputs/tables/selects\n"
    "- Tailwind CSS only — never inline styles or <style> blocks\n"
    "- $fetch with useRuntimeConfig().public.apiBase — never hard-coded URLs or mock services\n"
    "- Authorization: Bearer token + tenant_id from auth store on every API call\n"
    "- Zod schemas mirror backend Pydantic models exactly\n"
    "- Loading state: ref(false) + Skeleton loaders + empty state on every data page\n"
    "- try/catch/finally + Toast notifications on every API call\n"
    "- definePageMeta({ middleware: ['auth'] }) on every page except login\n"
    '- Bilingual fields: name_en + name_ar with dir="rtl" for Arabic inputs\n'
    "- <think> reasoning before every code response\n"
    "\n"
    "When given backend API endpoints, generate the matching frontend module.\n"
    "\n"
    "Output Format:\n"
    "- Produce exactly 6 files per module\n"
    "- Start each file with: ### FILE: <relative_path>\n"
    "- Wrap code in fenced blocks (```typescript or ```vue)\n"
    "- Separate files with --- on its own line\n"
    "- Files: types.ts, composable, list page, form page, edit page, detail page"
)


# =============================================================================
# Architecture labels & file ordering (exact match to training data)
# =============================================================================

ARCH_LABELS = {
    "service_layer": "Service Layer",
    "ddd": "DDD (Domain-Driven Design)",
    "cqrs": "CQRS (Command Query Responsibility Segregation)",
    "hexagonal": "Hexagonal (Ports and Adapters)",
    "event_sourced": "Event Sourcing",
}

FILE_ORDER = {
    "service_layer": [
        "models.py",
        "exceptions.py",
        "service.py",
        "schemas.py",
        "router.py",
    ],
    "ddd": [
        "domain/value_objects.py",
        "domain/models.py",
        "domain/exceptions.py",
        "infrastructure/models.py",
        "infrastructure/repository.py",
        "application/commands.py",
        "application/queries.py",
        "api/schemas.py",
        "api/router.py",
    ],
    "cqrs": [
        "exceptions.py",
        "write_models.py",
        "read_models.py",
        "schemas.py",
        "handlers.py",
        "router.py",
    ],
    "hexagonal": [
        "domain/models.py",
        "domain/exceptions.py",
        "domain/ports.py",
        "ports/inbound.py",
        "ports/outbound.py",
        "infrastructure/models.py",
        "infrastructure/repository.py",
        "adapters/persistence/orm.py",
        "adapters/persistence/repository.py",
        "application/service.py",
        "adapters/api/schemas.py",
        "adapters/api/router.py",
        "dependencies.py",
        "router.py",
        "schemas.py",
    ],
    "event_sourced": [
        "domain/value_objects.py",
        "domain/aggregates.py",
        "domain/events.py",
        "domain/exceptions.py",
        "infrastructure/models.py",
        "infrastructure/read_models.py",
        "infrastructure/repository.py",
        "application/commands.py",
        "application/queries.py",
        "api/schemas.py",
        "api/router.py",
    ],
}

# Module registry: maps module name → architecture pattern (45 modules)
MODULE_REGISTRY = {
    "crm": "service_layer", "fleet": "service_layer", "hr": "service_layer",
    "hr_attendance": "service_layer", "hr_recruitment": "service_layer",
    "hr_timesheet": "service_layer", "loyalty": "service_layer",
    "maintenance": "service_layer", "payment": "service_layer",
    "quality_control": "service_layer", "rating": "service_layer",
    "repair": "service_layer", "resource": "service_layer", "uom": "service_layer",
    "calendar_module": "service_layer", "gamification": "service_layer",
    "account": "ddd", "analytic": "ddd", "sale": "ddd", "purchase": "ddd",
    "stock": "ddd", "fixed_assets": "ddd", "project": "ddd", "hr_expense": "ddd",
    "tax": "ddd", "bom": "ddd", "approval": "ddd", "event": "ddd",
    "hr_holidays": "ddd", "mass_mailing": "ddd", "point_of_sale": "ddd",
    "purchase_requisition": "ddd", "sign": "ddd", "subscription": "ddd",
    "survey": "ddd",
    "stock_landed_costs": "cqrs", "bank_reconciliation": "cqrs",
    "helpdesk": "cqrs", "discuss": "cqrs",
    "warehouse": "hexagonal", "documents": "hexagonal", "website": "hexagonal",
    "mrp": "event_sourced",
}

# Module descriptions for prompt construction
MODULE_DESCRIPTIONS = {
    "crm": {"name": "CRM (Customer Relationship Management)", "entities": "leads, opportunities, customers, pipeline stages", "domain": "sales pipeline tracking, lead scoring, customer management"},
    "fleet": {"name": "Fleet Management", "entities": "vehicles, service logs, fuel entries", "domain": "company vehicle tracking, maintenance scheduling, fuel monitoring"},
    "hr": {"name": "Human Resources", "entities": "employees, departments, job positions, contracts", "domain": "employee lifecycle, organizational structure, contract management"},
    "hr_attendance": {"name": "HR Attendance", "entities": "attendance records, check-in/check-out events, overtime", "domain": "employee time tracking, presence monitoring, overtime calculation"},
    "hr_recruitment": {"name": "HR Recruitment", "entities": "applicants, job openings, interview stages, recruitment pipeline", "domain": "talent acquisition, application tracking, hiring workflow"},
    "hr_timesheet": {"name": "HR Timesheet", "entities": "timesheet entries, analytic lines, project hour logging", "domain": "work hour tracking, project time allocation, billable hours"},
    "loyalty": {"name": "Loyalty Program", "entities": "loyalty programs, rules, rewards, customer points", "domain": "customer retention, point accumulation, reward redemption"},
    "maintenance": {"name": "Maintenance Management", "entities": "maintenance requests, equipment, intervention types", "domain": "equipment upkeep, preventive maintenance, repair tracking"},
    "payment": {"name": "Payment Processing", "entities": "payment transactions, payment methods, payment providers", "domain": "transaction processing, multi-provider payment handling"},
    "quality_control": {"name": "Quality Control", "entities": "quality checks, inspection points, quality alerts", "domain": "product quality assurance, inspection workflows, defect tracking"},
    "rating": {"name": "Rating System", "entities": "ratings, feedback, satisfaction scores", "domain": "customer satisfaction tracking, service rating collection"},
    "repair": {"name": "Repair Orders", "entities": "repair orders, repair lines, parts used", "domain": "product repair workflow, parts tracking, repair invoicing"},
    "resource": {"name": "Resource Management", "entities": "resources, resource calendars, availability slots", "domain": "shared resource scheduling, availability management"},
    "uom": {"name": "Unit of Measure", "entities": "UoM categories, units, conversion factors", "domain": "measurement unit definition, cross-unit conversion"},
    "calendar_module": {"name": "Calendar Events", "entities": "calendar events, attendees, recurrence rules", "domain": "event scheduling, attendee management, recurring events"},
    "gamification": {"name": "Gamification", "entities": "challenges, goals, badges, user progress", "domain": "employee engagement, achievement tracking, badge rewards"},
    "account": {"name": "Accounting", "entities": "accounts, journal entries, journal items, fiscal years", "domain": "double-entry bookkeeping, chart of accounts, period closing"},
    "analytic": {"name": "Analytic Accounting", "entities": "analytic accounts, analytic lines, cost centers", "domain": "cost allocation, revenue tracking, cross-department analysis"},
    "sale": {"name": "Sales Orders", "entities": "sale orders, order lines, pricing rules", "domain": "sales workflow (draft -> confirmed -> done), discounting, taxes"},
    "purchase": {"name": "Purchase Orders", "entities": "purchase orders, order lines, vendor management", "domain": "procurement workflow, vendor selection, purchase approval"},
    "stock": {"name": "Inventory / Stock", "entities": "stock moves, stock quants, stock locations, warehouses", "domain": "inventory tracking, stock movements, location management"},
    "fixed_assets": {"name": "Fixed Assets", "entities": "assets, depreciation schedules, asset categories", "domain": "asset lifecycle, depreciation calculation, disposal tracking"},
    "project": {"name": "Project Management", "entities": "projects, tasks, milestones, task stages", "domain": "project planning, task assignment, progress tracking"},
    "hr_expense": {"name": "HR Expenses", "entities": "expense sheets, expense lines, expense categories", "domain": "employee expense reporting, approval workflow, reimbursement"},
    "tax": {"name": "Tax / Fiscal", "entities": "tax groups, tax rates, fiscal positions", "domain": "tax computation, multi-rate support, fiscal position mapping"},
    "bom": {"name": "Bill of Materials", "entities": "BOMs, BOM lines, BOM operations, work centers", "domain": "product composition, manufacturing routing, work center capacity"},
    "approval": {"name": "Approvals", "entities": "approval requests, approval categories, approvers", "domain": "multi-level approval workflow, request routing, deadline tracking"},
    "event": {"name": "Events", "entities": "events, registrations, event categories, tickets", "domain": "event organization, attendee registration, capacity management"},
    "hr_holidays": {"name": "HR Leave / Holidays", "entities": "leave requests, leave allocations, leave types", "domain": "employee time-off management, approval workflow, balance tracking"},
    "mass_mailing": {"name": "Mass Mailing", "entities": "mailing campaigns, mailing contacts, mailing lists", "domain": "email campaign management, contact segmentation, delivery tracking"},
    "point_of_sale": {"name": "Point of Sale", "entities": "POS orders, POS order lines, POS sessions, POS configs", "domain": "retail transaction processing, session management, cash control"},
    "purchase_requisition": {"name": "Purchase Requisitions", "entities": "requisitions, requisition lines, vendor proposals", "domain": "procurement requests, multi-vendor bidding, requisition approval"},
    "sign": {"name": "Electronic Signatures", "entities": "sign requests, sign templates, sign items, signers", "domain": "document signing workflow, template management, signer tracking"},
    "subscription": {"name": "Subscriptions", "entities": "subscriptions, subscription lines, subscription templates", "domain": "recurring billing, subscription lifecycle, renewal management"},
    "survey": {"name": "Surveys", "entities": "surveys, questions, answer options, user responses", "domain": "survey creation, response collection, results analysis"},
    "stock_landed_costs": {"name": "Stock Landed Costs", "entities": "landed cost records, cost lines, valuation adjustments", "domain": "import cost allocation, inventory valuation adjustment"},
    "bank_reconciliation": {"name": "Bank Reconciliation", "entities": "bank statements, statement lines, reconciliation matches", "domain": "bank statement import, transaction matching, balance reconciliation"},
    "helpdesk": {"name": "Helpdesk", "entities": "tickets, teams, SLA policies", "domain": "support ticket management, SLA tracking, team assignment"},
    "discuss": {"name": "Discussion / Messaging", "entities": "channels, messages, message reactions", "domain": "internal messaging, channel-based communication, real-time chat"},
    "warehouse": {"name": "Warehouse Management", "entities": "warehouses, zones, bins, stock locations", "domain": "warehouse layout, zone management, bin assignment, stock allocation"},
    "documents": {"name": "Document Management", "entities": "documents, folders, tags, document shares", "domain": "file organization, tagging, folder hierarchy, access control"},
    "website": {"name": "Website / CMS", "entities": "pages, menus, website configuration, page elements", "domain": "content management, page publishing, navigation structure"},
    "mrp": {"name": "Manufacturing (MRP)", "entities": "manufacturing orders, work orders, production events", "domain": "production planning, work order execution, event-driven state tracking"},
}


# =============================================================================
# Turn-1 prompt templates (match training data variations)
# =============================================================================

TURN1_TEMPLATES = [
    "Create a {arch_label} module for {module_name}. "
    "The module should manage {entities}.",

    "Build an ERP module for {domain}. "
    "Use {arch_label} architecture with proper layer separation.",

    "Create a {arch_label} module for {module_name} with bilingual support "
    "(English and Arabic fields). It should handle {entities}.",
]

# Subsequent-turn prompt templates keyed by file path
SUBSEQUENT_TURN_TEMPLATES = {
    "domain/value_objects.py": "Define the domain value objects for this module.",
    "domain/models.py": "Create the domain models (entities and aggregate roots) for this module.",
    "domain/exceptions.py": "Define the domain exceptions for this module.",
    "infrastructure/models.py": "Now create the SQLAlchemy ORM models that map to database tables.",
    "infrastructure/repository.py": "Create the repository that maps between domain models and ORM models.",
    "application/commands.py": "Create the application command handlers (create, update, delete operations).",
    "application/queries.py": "Create the query handlers for fetching data.",
    "api/schemas.py": "Create the Pydantic API schemas for requests and responses.",
    "api/router.py": "Create the FastAPI router with all REST endpoints.",
    "models.py": "Create the SQLAlchemy ORM models for this module.",
    "exceptions.py": "Define the custom exceptions for this module.",
    "service.py": "Create the service layer with business logic operations.",
    "schemas.py": "Create the Pydantic schemas for API request/response validation.",
    "router.py": "Create the FastAPI router with REST endpoints.",
    "write_models.py": "Create the write-side ORM models for command operations.",
    "read_models.py": "Create the denormalized read models for query operations.",
    "handlers.py": "Create the CQRS command and query handlers.",
    "domain/ports.py": "Define the ports (interfaces) for the domain layer.",
    "ports/inbound.py": "Define the inbound ports (use case interfaces) for this module.",
    "ports/outbound.py": "Define the outbound ports (repository interfaces) for this module.",
    "adapters/persistence/orm.py": "Create the persistence adapter ORM models.",
    "adapters/persistence/repository.py": "Implement the persistence adapter repository.",
    "application/service.py": "Create the application service implementing the inbound ports.",
    "adapters/api/schemas.py": "Create the API adapter schemas for this module.",
    "adapters/api/router.py": "Create the API adapter router with REST endpoints.",
    "dependencies.py": "Create the dependency injection wiring for this module.",
    "domain/aggregates.py": "Create the domain aggregates with event-driven state changes.",
    "domain/events.py": "Define the domain events emitted by the aggregates.",
    "infrastructure/read_models.py": "Create the denormalized read models for event-sourced queries.",
}


# =============================================================================
# Prompt builders
# =============================================================================

def build_backend_turn1_prompt(module_name: str, arch: str, variant: int = 0) -> str:
    desc = MODULE_DESCRIPTIONS.get(module_name)
    if not desc:
        raise ValueError(f"Unknown module: {module_name}. Add it to MODULE_DESCRIPTIONS.")

    arch_label = ARCH_LABELS.get(arch)
    if not arch_label:
        raise ValueError(f"Unknown architecture: {arch}")

    template = TURN1_TEMPLATES[variant % len(TURN1_TEMPLATES)]
    return template.format(
        arch_label=arch_label,
        module_name=desc["name"],
        entities=desc["entities"],
        domain=desc["domain"],
    )


def build_backend_subsequent_prompt(file_path: str) -> str:
    return SUBSEQUENT_TURN_TEMPLATES.get(
        file_path,
        f"Now create the {file_path} file for this module.",
    )


def build_frontend_prompt(
    module_name: str,
    api_prefix: str,
    entity: str,
    entity_plural: str,
    fields: list[dict],
    statuses: Optional[dict] = None,
    has_lines: bool = False,
    line_fields: Optional[list[dict]] = None,
    line_entity: Optional[str] = None,
) -> str:
    field_lines = []
    for f in fields:
        req = "required" if f.get("required") else "optional"
        field_lines.append(f"  - {f['name']}: {f['ts']} ({req}) [{f['component']}]")

    parts = [
        f"Generate the complete Nuxt 3 frontend module for **{entity_plural}**.",
        "",
        f"Backend API: `{api_prefix}`",
        f"Entity: `{entity}` (plural: `{entity_plural}`)",
        "",
        "Fields:",
        *field_lines,
    ]

    if statuses:
        status_str = ", ".join(f"{k} ({v})" for k, v in statuses.items())
        parts.extend(["", f"Statuses: {status_str}"])

    if has_lines and line_fields:
        parts.extend(["", f"Line items ({line_entity}):"])
        for lf in line_fields:
            parts.append(f"  - {lf['name']}: {lf['ts']} [{lf['component']}]")

    entity_lower = entity[0].lower() + entity[1:] if entity else module_name
    entity_path = entity_lower.replace(" ", "-")

    parts.extend([
        "",
        "Generate exactly 6 files in this order, each preceded by "
        "`### FILE:` header and separated by `---`:",
        f"1. ### FILE: types.ts",
        f"2. ### FILE: composables/use{entity}.ts",
        f"3. ### FILE: pages/{entity_path}/index.vue        (list page)",
        f"4. ### FILE: pages/{entity_path}/create.vue       (form/create page)",
        f"5. ### FILE: pages/{entity_path}/[id]/edit.vue    (edit page)",
        f"6. ### FILE: pages/{entity_path}/[id]/index.vue   (detail page)",
    ])

    return "\n".join(parts)


def build_frontend_fallback_prompt(module_name: str) -> str:
    """Structured fallback when module_def extraction fails."""
    desc = MODULE_DESCRIPTIONS.get(module_name, {})
    mod_name = desc.get("name", module_name.replace("_", " ").title())
    entities_str = desc.get("entities", module_name)
    domain_str = desc.get("domain", "")

    entity = module_name.replace("_", " ").title().replace(" ", "")
    entity_lower = entity[0].lower() + entity[1:]

    parts = [
        f"Generate the complete Nuxt 3 frontend module for **{mod_name}**.",
        "",
        f"Backend API: `/api/{module_name}`",
        f"Entity: `{entity}` (plural: `{entity}s`)",
        f"Domain: {domain_str}" if domain_str else "",
        f"It manages: {entities_str}.",
        "",
        "Default fields (infer additional fields from the domain):",
        "  - name_en: string (required) [InputText]",
        "  - name_ar: string (optional) [InputText]",
        "  - description: string (optional) [Textarea]",
        "  - is_active: boolean (optional) [Checkbox]",
        "",
        "Generate exactly 6 files in this order, each preceded by "
        "`### FILE:` header and separated by `---`:",
        f"1. ### FILE: types.ts",
        f"2. ### FILE: composables/use{entity}.ts",
        f"3. ### FILE: pages/{entity_lower}/index.vue        (list page)",
        f"4. ### FILE: pages/{entity_lower}/create.vue       (form/create page)",
        f"5. ### FILE: pages/{entity_lower}/[id]/edit.vue    (edit page)",
        f"6. ### FILE: pages/{entity_lower}/[id]/index.vue   (detail page)",
    ]
    return "\n".join(p for p in parts if p is not None)
