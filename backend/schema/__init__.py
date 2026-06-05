"""
Schema subpackage for workflow definition and validation.

What it does:
    Defines the data models for workflow schemas including nodes, edges, and
    configuration. Provides persistence to YAML files and structural validation
    of workflow graphs.

Entities in it:
    - models: Pydantic models for WorkflowSchema, NodeDefinition, EdgeDefinition,
      and all supporting configuration types and enumerations.
    - persistence: SchemaPersistence class for YAML-based save/load/list/delete.
    - validation: SchemaValidator for graph integrity, type compatibility, cycle
      detection, and reachability analysis.

How used by other modules:
    The frontend uses models to construct workflow schemas in the visual editor.
    The orchestration engine loads schemas via persistence and validates them
    before execution. The agent subpackage reads NodeConfig from schema nodes
    to configure individual agents.
"""

from backend.schema.models import (
    NodeType,
    EdgeType,
    LoggingLevel,
    GroupStructure,
    WorkflowConfig,
    NodeConfig,
    AgentGroupConfig,
    NodePosition,
    NodeDefinition,
    EdgeDefinition,
    WorkflowSchema,
)
from backend.schema.persistence import SchemaPersistence
from backend.schema.validation import SchemaValidator, SchemaValidationError
