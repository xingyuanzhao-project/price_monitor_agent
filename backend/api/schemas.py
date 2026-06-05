"""
API router for workflow schema CRUD and validation.

What it does:
    Exposes REST endpoints to list, create, read, update, delete, and
    validate workflow schemas.  All persistence is delegated to a
    SchemaPersistence instance and structural validation to a
    SchemaValidator instance, both injected via ``init``.

Entities in it:
    - router: FastAPI APIRouter mounted at ``/api/schemas``.
    - init: Dependency-injection entry point called at application startup.

How used by other modules:
    ``backend.main`` calls ``init(persistence, validator)`` during lifespan
    startup and includes the returned router in the FastAPI app.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.schema.models import WorkflowSchema
from backend.schema.persistence import SchemaPersistence
from backend.schema.validation import SchemaValidator, SchemaValidationError

router = APIRouter(prefix="/api/schemas", tags=["schemas"])

_persistence: SchemaPersistence | None = None
_validator: SchemaValidator | None = None


def init(
    persistence: SchemaPersistence,
    validator: SchemaValidator,
) -> APIRouter:
    """Wire backend dependencies and return the configured router.

    Args:
        persistence: SchemaPersistence instance for YAML read/write.
        validator: SchemaValidator for graph-integrity checks.

    Returns:
        The fully configured APIRouter.
    """
    global _persistence, _validator  # noqa: PLW0603
    _persistence = persistence
    _validator = validator
    return router


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/")
@router.get("")
async def list_schemas() -> list[dict]:
    """List all stored workflow schemas with summary info.

    Returns:
        list[dict]: Schema summaries with schema_id, name, description.
    """
    schema_ids = _persistence.list_schemas()
    summaries = []
    for schema_id in schema_ids:
        schema = _persistence.load_schema(schema_id)
        summaries.append({
            "schema_id": schema.schema_id,
            "name": schema.name,
            "description": schema.description,
        })
    return summaries


@router.post("/", status_code=201)
async def create_schema(schema: WorkflowSchema) -> WorkflowSchema:
    """Create (or overwrite) a workflow schema.

    Args:
        schema: The WorkflowSchema to persist.

    Returns:
        The persisted WorkflowSchema.
    """
    _persistence.save_schema(schema)
    return schema


@router.get("/{schema_id}")
async def get_schema(schema_id: str) -> WorkflowSchema:
    """Retrieve a single workflow schema by ID.

    Args:
        schema_id: Unique schema identifier.

    Returns:
        The matching WorkflowSchema.

    Raises:
        HTTPException 404: If the schema does not exist.
    """
    try:
        return _persistence.load_schema(schema_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"Schema '{schema_id}' not found")


@router.put("/{schema_id}")
async def update_schema(
    schema_id: str,
    schema: WorkflowSchema,
) -> WorkflowSchema:
    """Update an existing workflow schema.

    Args:
        schema_id: ID of the schema to replace.
        schema: The updated WorkflowSchema body.

    Returns:
        The persisted WorkflowSchema.

    Raises:
        HTTPException 404: If the original schema does not exist.
    """
    try:
        _persistence.load_schema(schema_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"Schema '{schema_id}' not found")
    _persistence.save_schema(schema)
    return schema


@router.delete("/{schema_id}")
async def delete_schema(schema_id: str) -> dict[str, str]:
    """Delete a workflow schema.

    Args:
        schema_id: ID of the schema to remove.

    Returns:
        ``{"status": "deleted"}``.

    Raises:
        HTTPException 404: If the schema does not exist.
    """
    try:
        _persistence.delete_schema(schema_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"Schema '{schema_id}' not found")
    return {"status": "deleted"}


@router.post("/validate")
async def validate_schema_body(schema: WorkflowSchema) -> dict:
    """Run structural validation on a schema provided in the request body.

    This endpoint accepts the schema directly in the POST body, enabling
    the frontend to validate unsaved in-progress schemas.

    Args:
        schema: The WorkflowSchema to validate.

    Returns:
        ``{"valid": True, "errors": []}`` on success or
        ``{"valid": False, "errors": [...]}`` on failure.
    """
    try:
        _validator.validate(schema)
        return {"valid": True, "errors": []}
    except SchemaValidationError as exc:
        return {"valid": False, "errors": list(exc.errors)}


@router.post("/{schema_id}/validate")
async def validate_schema_by_id(schema_id: str) -> dict[str, str]:
    """Run structural validation on a stored schema.

    Args:
        schema_id: ID of the schema to validate.

    Returns:
        ``{"status": "valid"}`` on success.

    Raises:
        HTTPException 404: If the schema does not exist.
        HTTPException 422: If the schema fails validation.
    """
    try:
        schema = _persistence.load_schema(schema_id)
    except FileNotFoundError:
        raise HTTPException(404, detail=f"Schema '{schema_id}' not found")

    try:
        _validator.validate(schema)
    except SchemaValidationError as exc:
        raise HTTPException(
            422, detail={"validation_errors": list(exc.errors)}
        )

    return {"status": "valid"}
