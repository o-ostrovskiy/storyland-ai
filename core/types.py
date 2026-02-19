"""
Configuration types for the core SDK.

Decouples WorkflowExecutor from common.config so the backend can pass
its own configuration without reading from environment variables.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecutorConfig:
    """Configuration for WorkflowExecutor.

    The HTTP API layer can build this from common.config.Config.
    A backend can construct it directly from its own config source.
    """

    model_name: str
    google_api_key: str
    workflow_timeout: int = 300
    database_url: Optional[str] = None
    use_database: bool = False
    langfuse_secret_key: Optional[str] = None
    langfuse_public_key: Optional[str] = None
    langfuse_host: Optional[str] = None

    @classmethod
    def from_config(cls, config) -> "ExecutorConfig":
        """Build from a common.config.Config instance."""
        return cls(
            model_name=config.model_name,
            google_api_key=config.google_api_key,
            workflow_timeout=config.workflow_timeout,
            database_url=config.database_url,
            use_database=config.use_database,
            langfuse_secret_key=config.langfuse_secret_key,
            langfuse_public_key=config.langfuse_public_key,
            langfuse_host=config.langfuse_host,
        )
