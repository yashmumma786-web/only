from pydantic import BaseModel
from typing import Optional
class RecordSiblingVoteResponse(BaseModel):
    ok: bool
    message: Optional[str] = None