from pydantic import BaseModel, Field
from typing import Literal

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