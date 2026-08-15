import asyncio
import logging
import re
from typing import Annotated
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)

from app.chat.attachments import (
    MAX_FILE_SIZE,
    AttachmentProcessingError,
    AttachmentService,
)
from app.chat.store import ConversationStore
from app.schemas import (
    AttachmentUploadResponse,
    ConversationCreate,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationSummary,
    ConversationUpdate,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["Conversations"],
)

CLIENT_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:-]{8,128}$"
)


# ============================================================
# Dependencies
# ============================================================

def get_chat_store(
    request: Request,
) -> ConversationStore:
    store = getattr(
        request.app.state,
        "chat_store",
        None,
    )

    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation store is not available.",
        )

    return store


def get_attachment_service(
    request: Request,
) -> AttachmentService:
    service = getattr(
        request.app.state,
        "attachment_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Attachment service is not available.",
        )

    return service


def get_owner_id(
    x_client_id: Annotated[
        str | None,
        Header(alias="X-Client-ID"),
    ] = None,
) -> str:
    if not x_client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Client-ID header is required.",
        )

    owner_id = x_client_id.strip()

    if not CLIENT_ID_PATTERN.fullmatch(owner_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Client-ID header is invalid.",
        )

    return owner_id


OwnerId = Annotated[
    str,
    Depends(get_owner_id),
]

StoreDependency = Annotated[
    ConversationStore,
    Depends(get_chat_store),
]

AttachmentServiceDependency = Annotated[
    AttachmentService,
    Depends(get_attachment_service),
]


# ============================================================
# Helpers
# ============================================================

async def require_conversation(
    store: ConversationStore,
    owner_id: str,
    conversation_id: UUID,
) -> dict:
    conversation = await asyncio.to_thread(
        store.get_conversation,
        owner_id,
        conversation_id,
    )

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return conversation


# ============================================================
# Create conversation
# ============================================================

@router.post(
    "",
    response_model=ConversationSummary,
    status_code=status.HTTP_201_CREATED,
)
async def create_conversation(
    payload: ConversationCreate,
    owner_id: OwnerId,
    store: StoreDependency,
) -> ConversationSummary:
    conversation = await asyncio.to_thread(
        store.create_conversation,
        owner_id,
        payload.title,
    )

    return ConversationSummary.model_validate(
        conversation
    )


# ============================================================
# List conversations
# ============================================================

@router.get(
    "",
    response_model=ConversationListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_conversations(
    owner_id: OwnerId,
    store: StoreDependency,
) -> ConversationListResponse:
    conversations = await asyncio.to_thread(
        store.list_conversations,
        owner_id,
    )

    return ConversationListResponse(
        items=[
            ConversationSummary.model_validate(item)
            for item in conversations
        ]
    )


# ============================================================
# Conversation detail
# ============================================================

@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
    status_code=status.HTTP_200_OK,
)
async def get_conversation(
    conversation_id: UUID,
    owner_id: OwnerId,
    store: StoreDependency,
) -> ConversationDetailResponse:
    detail = await asyncio.to_thread(
        store.get_conversation_detail,
        owner_id,
        conversation_id,
    )

    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return ConversationDetailResponse.model_validate(
        detail
    )


# ============================================================
# Rename conversation
# ============================================================

@router.patch(
    "/{conversation_id}",
    response_model=ConversationSummary,
    status_code=status.HTTP_200_OK,
)
async def update_conversation(
    conversation_id: UUID,
    payload: ConversationUpdate,
    owner_id: OwnerId,
    store: StoreDependency,
) -> ConversationSummary:
    updated = await asyncio.to_thread(
        store.update_conversation_title,
        owner_id,
        conversation_id,
        payload.title,
    )

    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return ConversationSummary.model_validate(updated)


# ============================================================
# Delete conversation
# ============================================================

@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: UUID,
    owner_id: OwnerId,
    store: StoreDependency,
) -> Response:
    deleted = await asyncio.to_thread(
        store.delete_conversation,
        owner_id,
        conversation_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )


# ============================================================
# Upload attachment
# ============================================================

@router.post(
    "/{conversation_id}/attachments",
    response_model=AttachmentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_attachment(
    conversation_id: UUID,
    owner_id: OwnerId,
    store: StoreDependency,
    attachment_service: AttachmentServiceDependency,
    file: Annotated[UploadFile, File(...)],
) -> AttachmentUploadResponse:
    await require_conversation(
        store,
        owner_id,
        conversation_id,
    )

    try:
        data = await file.read(
            MAX_FILE_SIZE + 1
        )
    finally:
        await file.close()

    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "Kích thước file không được "
                "vượt quá 20 MB."
            ),
        )

    filename = file.filename or "document"
    mime_type = (
        file.content_type or ""
    ).lower().strip()

    try:
        attachment = await asyncio.to_thread(
            attachment_service.ingest,
            owner_id,
            conversation_id,
            filename,
            mime_type,
            data,
        )

    except AttachmentProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "Attachment runtime error"
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected attachment processing error"
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Không thể xử lý tài liệu tải lên."
            ),
        ) from exc

    return AttachmentUploadResponse.model_validate(
        {
            "attachment": attachment,
        }
    )


# ============================================================
# Delete attachment
# ============================================================

@router.delete(
    "/{conversation_id}/attachments/{attachment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_attachment(
    conversation_id: UUID,
    attachment_id: UUID,
    owner_id: OwnerId,
    store: StoreDependency,
) -> Response:
    await require_conversation(
        store,
        owner_id,
        conversation_id,
    )

    deleted = await asyncio.to_thread(
        store.delete_attachment,
        owner_id,
        conversation_id,
        attachment_id,
    )

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Attachment not found.",
        )

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )