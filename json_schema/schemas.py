from pydantic import BaseModel
from typing import Literal


class Prompt(BaseModel):
    """A json schema for prompts"""
    prompt: str


class Parameter(BaseModel):
    """A json schema for function parameters"""
    type: Literal["number", "integer", "boolean", "string"]


class FuncDef(BaseModel):
    """A json schema for function definitions"""
    name: str
    description: str
    parameters: dict[str, Parameter]
    returns: Parameter
