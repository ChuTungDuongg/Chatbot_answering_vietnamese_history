import io
import re
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pymupdf
import pytesseract
from PIL import Image, ImageOps, UnidentifiedImageError

from app.chat.store import ConversationStore
from app.rag.retrieval import clean_text, match_norm
from app.services.rag_service import RAGService


MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_PDF_PAGES = 100
MAX_TEMPORARY_CHUNKS = 400

CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
MIN_EXTRACTED_PAGE_CHARS = 80

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
}

IMAGE_FORMAT_TO_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
}

ATTACHMENT_QUERY_TERMS = {
    "tai lieu",
    "file",
    "pdf",
    "anh",
    "hinh anh",
    "trang",
    "dinh kem",
    "van ban nay",
    "tai lieu nay",
    "noi dung nay",
    "theo tai lieu",
}


class AttachmentProcessingError(ValueError):
    pass


def sanitize_filename(filename: str | None) -> str:
    safe_name = Path(filename or "document").name
    safe_name = re.sub(r"[\x00-\x1f\x7f]", "", safe_name)
    safe_name = re.sub(r"\s+", " ", safe_name).strip()

    if not safe_name:
        return "document"

    return safe_name[:180]


def normalize_document_text(value: str) -> str:
    text = value.replace("\x00", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def contains_enough_text(text: str) -> bool:
    alphanumeric = re.sub(
        r"[^0-9A-Za-zÀ-ỹĐđ]",
        "",
        text,
    )

    return len(alphanumeric) >= MIN_EXTRACTED_PAGE_CHARS


def resize_for_ocr(
    image: Image.Image,
    max_dimension: int = 3000,
) -> Image.Image:
    width, height = image.size
    largest_dimension = max(width, height)

    if largest_dimension <= max_dimension:
        return image

    scale = max_dimension / largest_dimension

    resized_size = (
        max(1, int(width * scale)),
        max(1, int(height * scale)),
    )

    return image.resize(
        resized_size,
        Image.Resampling.LANCZOS,
    )


def run_ocr(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = resize_for_ocr(image)

    text = pytesseract.image_to_string(
        image,
        lang="vie+eng",
        config="--oem 3 --psm 6",
    )

    return normalize_document_text(text)


def split_page_into_chunks(
    text: str,
    filename: str,
    page_number: int,
) -> list[dict[str, Any]]:
    words = text.split()

    if not words:
        return []

    chunks: list[dict[str, Any]] = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(
            start + CHUNK_WORDS,
            len(words),
        )

        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append(
                {
                    "title": (
                        f"{filename} - trang {page_number}"
                    ),
                    "text": chunk_text,
                    "page_number": page_number,
                    "chunk_index": chunk_index,
                }
            )

        if end >= len(words):
            break

        start = max(
            end - CHUNK_OVERLAP_WORDS,
            start + 1,
        )

        chunk_index += 1

    return chunks


def extract_pdf_pages(
    data: bytes,
) -> list[tuple[int, str]]:
    if not data.startswith(b"%PDF-"):
        raise AttachmentProcessingError(
            "Nội dung file không phải PDF hợp lệ."
        )

    try:
        document = pymupdf.open(
            stream=data,
            filetype="pdf",
        )
    except Exception as exc:
        raise AttachmentProcessingError(
            "Không thể mở file PDF."
        ) from exc

    try:
        if document.needs_pass:
            raise AttachmentProcessingError(
                "Không hỗ trợ PDF được bảo vệ bằng mật khẩu."
            )

        if document.page_count == 0:
            raise AttachmentProcessingError(
                "PDF không có trang nào."
            )

        if document.page_count > MAX_PDF_PAGES:
            raise AttachmentProcessingError(
                f"PDF không được vượt quá "
                f"{MAX_PDF_PAGES} trang."
            )

        pages: list[tuple[int, str]] = []

        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            extracted_text = normalize_document_text(
                page.get_text(
                    "text",
                    sort=True,
                )
            )

            if not contains_enough_text(extracted_text):
                matrix = pymupdf.Matrix(2.0, 2.0)

                pixmap = page.get_pixmap(
                    matrix=matrix,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                )

                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples,
                )

                extracted_text = run_ocr(image)

            if extracted_text:
                pages.append(
                    (
                        page_index + 1,
                        extracted_text,
                    )
                )

        if not pages:
            raise AttachmentProcessingError(
                "Không trích xuất được nội dung chữ từ PDF."
            )

        return pages
    finally:
        document.close()


def extract_image_text(
    data: bytes,
    expected_mime_type: str,
) -> list[tuple[int, str]]:
    try:
        with Image.open(io.BytesIO(data)) as image:
            actual_format = (
                image.format or ""
            ).upper()

            actual_mime_type = IMAGE_FORMAT_TO_MIME.get(
                actual_format
            )

            if actual_mime_type is None:
                raise AttachmentProcessingError(
                    "Định dạng hình ảnh không được hỗ trợ."
                )

            if actual_mime_type != expected_mime_type:
                raise AttachmentProcessingError(
                    "MIME type không khớp nội dung hình ảnh."
                )

            image.load()
            image_copy = image.copy()

    except UnidentifiedImageError as exc:
        raise AttachmentProcessingError(
            "File tải lên không phải hình ảnh hợp lệ."
        ) from exc

    text = run_ocr(image_copy)

    if not contains_enough_text(text):
        raise AttachmentProcessingError(
            "Không tìm thấy đủ nội dung chữ trong hình ảnh."
        )

    return [(1, text)]


def is_attachment_question(question: str) -> bool:
    normalized_question = match_norm(question)

    return any(
        term in normalized_question
        for term in ATTACHMENT_QUERY_TERMS
    )


def minmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.size == 0:
        return values

    minimum = float(values.min())
    maximum = float(values.max())

    if maximum - minimum < 1e-8:
        return np.ones_like(values)

    return (values - minimum) / (maximum - minimum)


class AttachmentService:
    def __init__(
        self,
        store: ConversationStore,
        rag_service: RAGService,
    ):
        self.store = store
        self.rag_service = rag_service

    def _ensure_embedder(self) -> None:
        if self.rag_service.embedder is None:
            raise RuntimeError(
                "Embedding model is not loaded."
            )

    @staticmethod
    def validate_upload(
        filename: str,
        mime_type: str,
        data: bytes,
    ) -> None:
        if mime_type not in ALLOWED_MIME_TYPES:
            raise AttachmentProcessingError(
                "Chỉ hỗ trợ PDF, PNG, JPEG và WebP."
            )

        if not data:
            raise AttachmentProcessingError(
                "File tải lên bị rỗng."
            )

        if len(data) > MAX_FILE_SIZE:
            raise AttachmentProcessingError(
                "Kích thước file không được vượt quá 20 MB."
            )

        if not filename:
            raise AttachmentProcessingError(
                "Tên file không hợp lệ."
            )

    @staticmethod
    def extract_pages(
        mime_type: str,
        data: bytes,
    ) -> list[tuple[int, str]]:
        if mime_type == "application/pdf":
            return extract_pdf_pages(data)

        return extract_image_text(
            data,
            expected_mime_type=mime_type,
        )

    def build_chunks(
        self,
        attachment_id: str,
        filename: str,
        pages: list[tuple[int, str]],
    ) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []

        for page_number, page_text in pages:
            page_chunks = split_page_into_chunks(
                page_text,
                filename,
                page_number,
            )

            for item in page_chunks:
                chunk_index = item.pop("chunk_index")

                item["chunk_id"] = (
                    f"temp:{attachment_id}:"
                    f"{page_number}:{chunk_index}"
                )

                chunks.append(item)

        if not chunks:
            raise AttachmentProcessingError(
                "Không tạo được chunk từ tài liệu."
            )

        if len(chunks) > MAX_TEMPORARY_CHUNKS:
            raise AttachmentProcessingError(
                "Tài liệu tạo ra quá nhiều chunk."
            )

        return chunks

    def create_embeddings(
        self,
        chunks: list[dict[str, Any]],
    ) -> np.ndarray:
        self._ensure_embedder()

        passages = [
            "passage: "
            + clean_text(
                f"{chunk['title']}\n{chunk['text']}"
            )
            for chunk in chunks
        ]

        embeddings = self.rag_service.embedder.encode(
            passages,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def ingest(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        filename: str,
        mime_type: str,
        data: bytes,
    ) -> dict[str, Any]:
        filename = sanitize_filename(filename)
        mime_type = (mime_type or "").lower().strip()

        self.validate_upload(
            filename,
            mime_type,
            data,
        )

        attachment = self.store.create_attachment(
            owner_id=owner_id,
            conversation_id=conversation_id,
            filename=filename,
            mime_type=mime_type,
            size_bytes=len(data),
        )

        attachment_id = str(attachment["id"])

        try:
            pages = self.extract_pages(
                mime_type,
                data,
            )

            chunks = self.build_chunks(
                attachment_id,
                filename,
                pages,
            )

            embeddings = self.create_embeddings(chunks)

            if len(embeddings) != len(chunks):
                raise RuntimeError(
                    "Embedding count does not match chunk count."
                )

            stored_chunks: list[dict[str, Any]] = []

            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            ):
                stored_chunks.append(
                    {
                        **chunk,
                        "embedding": embedding,
                    }
                )

            self.store.replace_temporary_chunks(
                owner_id=owner_id,
                conversation_id=conversation_id,
                attachment_id=attachment_id,
                chunks=stored_chunks,
            )

            ready_attachment = (
                self.store.update_attachment_status(
                    owner_id=owner_id,
                    conversation_id=conversation_id,
                    attachment_id=attachment_id,
                    status="ready",
                    chunk_count=len(stored_chunks),
                )
            )

            if ready_attachment is None:
                raise RuntimeError(
                    "Attachment disappeared during processing."
                )

            return ready_attachment

        except Exception as exc:
            self.store.update_attachment_status(
                owner_id=owner_id,
                conversation_id=conversation_id,
                attachment_id=attachment_id,
                status="failed",
                chunk_count=0,
                error=str(exc)[:500],
            )

            raise


class TemporaryCorpusRetriever:
    def __init__(
        self,
        store: ConversationStore,
        rag_service: RAGService,
    ):
        self.store = store
        self.rag_service = rag_service

    def retrieve(
        self,
        owner_id: str,
        conversation_id: str | UUID,
        question: str,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        chunks = self.store.list_temporary_chunks(
            owner_id,
            conversation_id,
        )

        if not chunks:
            return []

        if self.rag_service.embedder is None:
            raise RuntimeError(
                "Embedding model is not loaded."
            )

        question = clean_text(question)

        query_embedding = (
            self.rag_service.embedder.encode(
                ["query: " + question],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )[0]

        embeddings = np.vstack(
            [chunk["embedding"] for chunk in chunks]
        ).astype(np.float32)

        dense_scores = embeddings @ query_embedding

        prefetch_k = min(
            max(top_k * 3, 12),
            len(chunks),
        )

        indices = np.argsort(
            dense_scores
        )[::-1][:prefetch_k]

        candidates: list[dict[str, Any]] = []

        for index in indices:
            chunk = dict(chunks[int(index)])
            chunk.pop("embedding", None)

            chunk["temporary_dense_score"] = float(
                dense_scores[int(index)]
            )

            candidates.append(chunk)

        if (
            self.rag_service.reranker is not None
            and candidates
        ):
            pairs = [
                (
                    question,
                    clean_text(
                        f"{chunk['title']}\n{chunk['text']}"
                    ),
                )
                for chunk in candidates
            ]

            reranker_scores = (
                self.rag_service.reranker.predict(
                    pairs,
                    batch_size=32,
                    show_progress_bar=False,
                )
            )

            reranker_scores = np.asarray(
                reranker_scores,
                dtype=np.float32,
            ).reshape(-1)

            reranker_normalized = minmax(
                reranker_scores
            )

            dense_normalized = minmax(
                np.asarray(
                    [
                        chunk["temporary_dense_score"]
                        for chunk in candidates
                    ],
                    dtype=np.float32,
                )
            )

            for index, chunk in enumerate(candidates):
                chunk["reranker_score"] = float(
                    reranker_scores[index]
                )

                chunk["final_retrieval_score"] = float(
                    0.82 * reranker_normalized[index]
                    + 0.18 * dense_normalized[index]
                )
        else:
            normalized = minmax(
                np.asarray(
                    [
                        chunk["temporary_dense_score"]
                        for chunk in candidates
                    ],
                    dtype=np.float32,
                )
            )

            for index, chunk in enumerate(candidates):
                chunk["reranker_score"] = None
                chunk["final_retrieval_score"] = float(
                    normalized[index]
                )

        for chunk in candidates:
            chunk["metadata_bonus"] = 0.0
            chunk["metadata_hits"] = []
            chunk["rrf_score"] = None
            chunk["source_kind"] = "attachment"

        candidates.sort(
            key=lambda chunk: chunk.get(
                "final_retrieval_score",
                0.0,
            ),
            reverse=True,
        )

        return candidates[:max(1, top_k)]


def merge_global_and_temporary_contexts(
    question: str,
    global_contexts: list[dict[str, Any]],
    temporary_contexts: list[dict[str, Any]],
    rag_service: RAGService,
    final_k: int = 6,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for source_kind, contexts in (
        ("history", global_contexts),
        ("attachment", temporary_contexts),
    ):
        for original in contexts:
            chunk = dict(original)
            chunk_id = str(chunk.get("chunk_id", ""))

            if not chunk_id or chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)
            chunk["source_kind"] = chunk.get(
                "source_kind",
                source_kind,
            )
            candidates.append(chunk)

    if not candidates:
        return []

    if rag_service.reranker is None:
        candidates.sort(
            key=lambda chunk: chunk.get(
                "final_retrieval_score",
                0.0,
            ),
            reverse=True,
        )

        return candidates[:final_k]

    pairs = [
        (
            clean_text(question),
            clean_text(
                f"{chunk.get('title', '')}\n"
                f"{chunk.get('text', '')}"
            ),
        )
        for chunk in candidates
    ]

    reranker_scores = rag_service.reranker.predict(
        pairs,
        batch_size=32,
        show_progress_bar=False,
    )

    reranker_scores = np.asarray(
        reranker_scores,
        dtype=np.float32,
    ).reshape(-1)

    reranker_normalized = minmax(reranker_scores)

    previous_scores = np.asarray(
        [
            float(
                chunk.get(
                    "final_retrieval_score",
                    0.0,
                )
                or 0.0
            )
            for chunk in candidates
        ],
        dtype=np.float32,
    )

    previous_normalized = minmax(previous_scores)
    attachment_question = is_attachment_question(question)

    for index, chunk in enumerate(candidates):
        attachment_bonus = 0.0

        if (
            attachment_question
            and chunk["source_kind"] == "attachment"
        ):
            attachment_bonus = 0.08

        chunk["reranker_score"] = float(
            reranker_scores[index]
        )

        chunk["final_retrieval_score"] = float(
            0.85 * reranker_normalized[index]
            + 0.15 * previous_normalized[index]
            + attachment_bonus
        )

    candidates.sort(
        key=lambda chunk: chunk[
            "final_retrieval_score"
        ],
        reverse=True,
    )

    return candidates[:max(1, final_k)]