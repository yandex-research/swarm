from pydantic import BaseModel, conint, StrictFloat


class LocalMetrics(BaseModel):
    step: conint(ge=0, strict=True)
    loss: StrictFloat
