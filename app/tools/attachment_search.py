from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.chat.attachments import TemporaryCorpusRetriever
from app.tools.registry import ToolExecutionContext


class SearchUploadedDocumentsInput(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=8, ge=1, le=20)


class SearchUploadedDocumentsTool:
    name = "search_uploaded_documents"
    description = (
        "Search PDF and image documents uploaded to the current conversation. "
        "Use it for questions about an attached file or when uploaded evidence may help."
    )
    input_schema = SearchUploadedDocumentsInput

    def __init__(self, retriever: TemporaryCorpusRetriever):
        self.retriever = retriever

    def run(self, arguments: SearchUploadedDocumentsInput) -> list[dict[str, Any]]:
        raise RuntimeError("Uploaded-document search requires conversation context.")

    def run_with_context(
        self,
        arguments: SearchUploadedDocumentsInput,
        context: ToolExecutionContext,
    ) -> list[dict[str, Any]]:
        if not context.owner_id or not context.conversation_id:
            raise RuntimeError("Uploaded-document search requires owner and conversation context.")

        chunks = self.retriever.retrieve(
            owner_id=context.owner_id,
            conversation_id=context.conversation_id,
            question=arguments.query,
            top_k=arguments.top_k,
        )

        return [
            {
                **chunk,
                "source_kind": "attachment",
            }
            for chunk in chunks
        ]
