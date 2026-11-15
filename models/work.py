"""
A *work* a journey starts from: a book, a film or a series (Movies epic, story 1).

Every entry into the agent service used to be a book (``book_title`` +
``author``). Movies need one more *kind* of work, not a second copy of the
pipeline, so the AI contract carries one ``WorkRef`` with a ``kind``.

Compatibility rule (ADR in docs/ARCHITECTURE.md): the book fields stay on the
wire. A request that sends only ``book_title`` + ``author`` means
``kind="book"`` and behaves byte-identically to before this module existed.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

WorkKind = Literal["book", "film", "series"]

DEFAULT_WORK_KIND: WorkKind = "book"

# Same bounds as the request's book_title / author (api/models.py
# MAX_TITLE_LENGTH / MAX_AUTHOR_LENGTH): a work's title and creator flow into
# the same prompts, so they get the same cost + prompt-injection-surface guard.
# A unit test pins these equal to the api constants.
MAX_WORK_TITLE_LENGTH = 200
MAX_WORK_CREATOR_LENGTH = 200

# A film or series year far outside this range is a typo or an injection
# attempt, never a real release year.
MIN_WORK_YEAR = 1800
MAX_WORK_YEAR = 2100


def _non_blank(value: str, field_name: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


class WorkRef(BaseModel):
    """One work a journey starts from.

    ``creator`` is the author for a book, the director for a film, and the
    creator/showrunner for a series.
    """

    kind: WorkKind = Field(default=DEFAULT_WORK_KIND, description="book, film or series")
    title: str = Field(
        min_length=1, max_length=MAX_WORK_TITLE_LENGTH, description="Title of the work"
    )
    creator: str = Field(
        min_length=1,
        max_length=MAX_WORK_CREATOR_LENGTH,
        description="Author (book), director (film) or creator/showrunner (series)",
    )
    year: Optional[int] = Field(
        default=None,
        ge=MIN_WORK_YEAR,
        le=MAX_WORK_YEAR,
        description="Release or publication year, when known",
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        return _non_blank(value, "work.title")

    @field_validator("creator")
    @classmethod
    def validate_creator(cls, value: str) -> str:
        return _non_blank(value, "work.creator")
