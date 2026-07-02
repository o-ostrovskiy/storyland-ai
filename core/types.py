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
    environment: str = "local"
    # Result cache settings (carried through from common.config.Config).
    # cache_enabled defaults True so an absent value never silently disables
    # caching (matches common.config.Config's CACHE_ENABLED default).
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400
    cache_max_entries: int = 500
    # Cache backend: "memory" (in-process, default for direct/library use and
    # tests) or "disk" (persistent SQLite on a docker volume, survives
    # restart/redeploy). The deployed app sets this to "disk" via CACHE_BACKEND;
    # the dataclass default stays "memory" so unit tests are hermetic.
    cache_backend: str = "memory"
    # On-disk directory for the "disk" backend (mounted on a docker volume in
    # prod). Ignored by the memory backend.
    cache_dir: Optional[str] = None
    # Bounded Gemini retry backoff (parity with the eval runner).
    retry_exp_base: float = 2.0
    retry_max_delay: float = 12.0
    retry_attempts: int = 4
    # Book-recommendation floor (parity with common.config.Config).
    rec_min_results: int = 3

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
            environment=config.environment,
            cache_enabled=getattr(config, "cache_enabled", True),
            cache_ttl_seconds=getattr(config, "cache_ttl_seconds", 86400),
            cache_max_entries=getattr(config, "cache_max_entries", 500),
            cache_backend=getattr(config, "cache_backend", "memory"),
            cache_dir=getattr(config, "cache_dir", None),
            retry_exp_base=getattr(config, "retry_exp_base", 2.0),
            retry_max_delay=getattr(config, "retry_max_delay", 12.0),
            retry_attempts=getattr(config, "retry_attempts", 4),
            rec_min_results=getattr(config, "rec_min_results", 3),
        )
