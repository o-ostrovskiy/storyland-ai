"""What discovery knows about a film or series (Movies epic story 4, ai-private#67).

A screen work answers three separate questions, never blended:

1. Where is the story set? ``story_locations`` — the narrative's real places
   (Paris for *Emily in Paris*). Empty for an invented world (Westeros).
2. Where was it filmed? ``filming_locations`` — where scenes were shot
   (Dubrovnik for King's Landing).
3. Where is the book set? Only when there is a source book, ``source_work``
   (Story 6 supplies the confirmed book; discovery reads it, never invents it).
"""

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from models.work import WorkRef

# Each list is a hint for the city/landmark researchers, not a journey: a
# handful of places is plenty, and the cap bounds what flows into prompts.
MAX_SCREEN_LOCATIONS = 20


class ScreenContext(BaseModel):
    story_locations: List[str] = Field(
        default_factory=list,
        description="Real places where the story takes place. Empty for an invented world.",
    )
    filming_locations: List[str] = Field(
        default_factory=list, description="Real places where scenes were filmed."
    )
    source_work: Optional[WorkRef] = Field(
        default=None,
        description="The book the film or series is based on, when one exists; never guessed.",
    )
    is_fictional_world: bool = Field(
        default=False,
        description="True when the story's world is invented (Westeros, Middle-earth, Pandora).",
    )

    @field_validator("story_locations", "filming_locations")
    @classmethod
    def _clean_places(cls, places: List[str]) -> List[str]:
        seen: List[str] = []
        for place in places:
            name = place.strip() if isinstance(place, str) else ""
            if name and name.casefold() not in {s.casefold() for s in seen}:
                seen.append(name)
        return seen[:MAX_SCREEN_LOCATIONS]

    @field_validator("source_work")
    @classmethod
    def _source_is_a_book(cls, work: Optional[WorkRef]) -> Optional[WorkRef]:
        # A film's source is a book or nothing here: a stage musical or another
        # film is not "the book behind it", and saying so would invent one.
        return work if work is None or work.kind == "book" else None

    @model_validator(mode="after")
    def _invented_world_has_no_real_setting(self) -> "ScreenContext":
        # The epic's rule: no invented "set in" places. A fictional world's
        # journey is built from filming locations and creator sites only.
        if self.is_fictional_world and self.story_locations:
            self.story_locations = []
        return self
