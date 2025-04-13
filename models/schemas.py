from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any

class ParameterSchema(BaseModel):
    name: str = Field(
        description="The name of the parameter",
        example="schema_name"
    )
    location: Literal["path", "query", "header", "body"] = Field(
        description="Where the parameter is located in the request",
        example="path"
    )
    required: bool = Field(
        description="Whether the parameter is required",
        default=True
    )
    description: Optional[str] = Field(
        description="Description of the parameter",
        default=None,
        example="The name of the schema to retrieve"
    )
    type: str = Field(
        description="The data type of the parameter",
        example="string"
    )
    example: Optional[Any] = Field(
        description="Example value for the parameter",
        default=None
    )

class EndpointSchema(BaseModel):
    url: str = Field(
        description="The URL endpoint path",
        pattern="^/.*",  # Must start with /
        example="/webslayer/schema/{schema_name}"
    )
    type: Literal["GET", "POST", "PUT", "DELETE"] = Field(
        description="The HTTP method type",
        example="GET"
    )   
    parameters: List[ParameterSchema] = Field(
        description="List of parameters available for this endpoint",
        default_factory=list
    )
    description: Optional[str] = Field(
        description="Description of what the endpoint does",
        default=None,
        example="Retrieves a specific schema by name"
    )