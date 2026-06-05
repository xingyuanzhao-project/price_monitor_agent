"""
API subpackage exposing FastAPI routers for the Price Monitor Agent.

What it does:
    Provides HTTP endpoints for schema CRUD, workflow run management
    (including SSE streaming), user-settings persistence, and LLM model /
    tool discovery.  Each module defines a single APIRouter and an ``init``
    function that wires in the required backend dependencies.

Entities in it:
    - schemas: CRUD + validation for WorkflowSchema resources.
    - runs: Start runs, stream events via SSE, list run records.
    - settings: Read / update user settings.
    - models: List LLM models per provider, list registered tools.

How used by other modules:
    ``backend.main`` calls each module's ``init`` function during the
    FastAPI lifespan and includes the returned routers in the application.
"""
