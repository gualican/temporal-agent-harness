# ABOUTME: The shared pydantic base for payload models that cross Temporal's data converter
# (workflow state, update/query params, activity args, child-workflow requests and results).

from pydantic import BaseModel, model_validator


class Model(BaseModel):
    """Base for payload models in code built on the harness.

    The OpenAI Agents plugin's payload converter serializes with
    exclude_unset=True (so OpenAI's drifting response models round-trip), which
    silently drops any field not in __pydantic_fields_set__ — and in-place
    mutations like list.append never mark a field as set. Workflow state that
    is mutated in place and then serialized (queries, update results, and
    especially continue-as-new arguments) would lose those fields. Marking all
    fields as set after validation makes exclude_unset a no-op for our models.
    """

    @model_validator(mode="after")
    def _mark_all_fields_set(self) -> "Model":
        self.__pydantic_fields_set__.update(type(self).model_fields)
        return self
