"""Upload/OCR and Central attachment scope using tiny images and fake providers only."""
from io import BytesIO
import asyncio
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from PIL import Image
from pydantic import ValidationError

from app.api.conversations import router
from app.api.routes import _execute_chat
from app.chat.attachments import AttachmentService, AttachmentProcessingError, OCRResult, TemporaryCorpusRetriever, MAX_FILE_SIZE
from app.chat.store import ConversationStore
from app.schemas import ChatRequest
from app.tools.attachment_search import SearchUploadedDocumentsTool
from app.agents.central.model_runtime import CentralGeneration
from tests.test_attachment_agent_flow import FakeEmbedder
from tests.test_central_agent import FakeTool, FakeCentralRuntime, build_agent


TEXT = "Tài liệu trình bày việc chuẩn bị lực lượng và tổ chức, huy động sự ủng hộ của nhân dân. Thời cơ và điều kiện chính trị có vai trò trong quá trình giành chính quyền."


class FakeOCR:
    name = "fake-ocr-cpu"

    def __init__(self, text=TEXT, error=False):
        self.text, self.error, self.calls = text, error, 0

    def extract(self, image):
        self.calls += 1
        assert image.size == (12, 12)
        if self.error:
            raise RuntimeError("C:/private/tesseract.exe is missing")
        return OCRResult(self.text, confidence=0.82)


def image_bytes(fmt="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (12, 12), "white").save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def setup(tmp_path):
    store = ConversationStore(tmp_path / "chat.sqlite")
    conversation = store.create_conversation("owner-123")
    rag = SimpleNamespace(embedder=FakeEmbedder(), reranker=None)
    provider = FakeOCR()
    service = AttachmentService(store, rag, ocr_provider=provider)
    app = FastAPI()  # No production lifespan / model initialization.
    app.include_router(router)
    app.state.chat_store, app.state.attachment_service = store, service
    class LocalClient:
        def post(self, path, *, headers, files, data):
            # Exercise the real ASGI multipart parser without a network server or HTTP client dependency.
            boundary = "fixture-multipart-boundary"
            parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode() for key, value in data.items()]
            for key, (name, content, mime) in files.items():
                parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"; filename="{name}"\r\nContent-Type: {mime}\r\n\r\n'.encode() + content + b"\r\n")
            body = b"".join(parts) + f"--{boundary}--\r\n".encode()
            async def request():
                events = []
                async def receive():
                    return {"type": "http.request", "body": body, "more_body": False}
                async def send(event):
                    events.append(event)
                scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST",
                    "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
                    "client": ("test", 1), "server": ("test", 80), "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()]
                    + [(b"content-type", f"multipart/form-data; boundary={boundary}".encode()), (b"content-length", str(len(body)).encode())]}
                await app(scope, receive, send)
                text = b"".join(event.get("body", b"") for event in events).decode()
                return SimpleNamespace(status_code=events[0]["status"], text=text, json=lambda: json.loads(text))
            return asyncio.run(request())
    yield store, conversation["id"], rag, provider, service, LocalClient()


@pytest.mark.parametrize("fmt,mime,extension", [("PNG", "image/png", "png"), ("JPEG", "image/jpeg", "jpg"), ("WEBP", "image/webp", "webp")])
def test_clipboard_and_picker_use_identical_multipart_ocr_path(setup, fmt, mime, extension):
    store, cid, rag, provider, service, client = setup
    for origin in ["clipboard", "file"]:
        response = client.post(f"/api/v1/conversations/{cid}/attachments", headers={"X-Client-ID": "owner-123"},
            files={"file": (f"clipboard-image-1.{extension}", image_bytes(fmt), mime)}, data={"upload_origin": origin})
        assert response.status_code == 201, response.text
        item = response.json()["attachment"]
        assert item["upload_origin"] == origin and item["status"] == "ready"
        assert item["ocr"]["ocr_success"] and item["ocr"]["ocr_provider"] == "fake-ocr-cpu"
        assert item["ocr"]["ocr_confidence"] == .82
    assert provider.calls == 2
    rows = TemporaryCorpusRetriever(store, rag).retrieve("owner-123", cid, "Đọc ảnh")
    assert {row["upload_origin"] for row in rows} == {"file", "clipboard"}
    assert all(row["text"] == TEXT and row["mime_type"] == mime for row in rows)


@pytest.mark.parametrize("provider", [FakeOCR(text=""), FakeOCR(error=True)])
def test_ocr_failure_is_controlled_not_citable_and_does_not_leak_local_paths(setup, provider):
    store, cid, rag, _, service, client = setup
    service.ocr_provider = provider
    response = client.post(f"/api/v1/conversations/{cid}/attachments", headers={"X-Client-ID": "owner-123"},
        files={"file": ("image.png", image_bytes(), "image/png")}, data={"upload_origin": "clipboard"})
    assert response.status_code == 422
    assert "private" not in response.text and "tesseract.exe" not in response.text
    assert store.list_temporary_chunks("owner-123", cid) == []
    failed = store.list_attachments("owner-123", cid)[0]
    assert failed["status"] == "failed" and not failed["ocr"]["ocr_success"]
    assert failed["ocr"]["ocr_error"]


def test_mime_contents_size_and_total_limit_are_validated_before_ocr(setup):
    store, cid, _, provider, service, _ = setup
    for mime, data in [("image/svg+xml", b"<svg/>"), ("image/png", b"not an image"), ("image/jpeg", image_bytes()), ("image/png", b"0" * (MAX_FILE_SIZE + 1))]:
        with pytest.raises(AttachmentProcessingError):
            service.ingest("owner-123", cid, "../../image.png", mime, data)
    assert provider.calls == 0
    for item in store.list_attachments("owner-123", cid):
        store.delete_attachment("owner-123", cid, item["id"])
    for i in range(5):
        service.ingest("owner-123", cid, f"C:\\private\\image-{i}.png", "image/png", image_bytes())
    assert all("private" not in item["filename"] for item in store.list_attachments("owner-123", cid))
    with pytest.raises(AttachmentProcessingError, match="tối đa 5"):
        service.ingest("owner-123", cid, "six.png", "image/png", image_bytes())
    assert provider.calls == 5


def test_central_image_only_uses_scoped_ocr_evidence_one_llm_and_keeps_user_text_empty(setup, monkeypatch):
    store, cid, rag, provider, service, _ = setup
    first = service.ingest("owner-123", cid, "clipboard-image-1.png", "image/png", image_bytes(), "clipboard")
    other = service.ingest("owner-123", cid, "other.png", "image/png", image_bytes())
    retriever = TemporaryCorpusRetriever(store, rag)
    assert retriever.retrieve("wrong-owner", cid, "ảnh") == []
    assert retriever.retrieve("owner-123", str(uuid4()), "ảnh") == []
    assert {row["attachment_id"] for row in retriever.retrieve("owner-123", cid, "ảnh", attachment_ids=(first["id"],))} == {first["id"]}
    runtime = FakeCentralRuntime([CentralGeneration(content="Theo ảnh đính kèm, tài liệu trình bày việc chuẩn bị lực lượng, tổ chức và huy động sự ủng hộ của nhân dân. [S1]")])
    agent = build_agent(runtime, FakeTool("search_history", []), SearchUploadedDocumentsTool(retriever), has_documents=lambda owner, conversation: True)
    monkeypatch.setattr("app.api.routes._gpu_name", lambda: None)
    payload = ChatRequest(conversation_id=cid, question="", attachment_ids=[first["id"]], mode="central")
    result = _execute_chat(store, agent, rag, "owner-123", payload, "fake-image-request", "central")
    assert result["status"] == "ok", result["central_debug"]
    messages = store.list_messages("owner-123", cid)
    assert messages[0]["content"] == ""
    assert messages[0]["sources"][0]["attachment_id"] == first["id"]
    assert len(runtime.calls) == 1
    prompt = runtime.calls[0]["messages"][-1]["content"]
    assert "ATTACHMENT" in prompt and "OCR" in prompt and "fake-ocr-cpu" in prompt
    assert "other.png" not in prompt
    debug = result["central_debug"]
    assert debug["attachment_count"] == debug["clipboard_image_count"] == debug["attachment_tool_calls"] == 1
    assert debug["ocr_used"] and debug["ocr_text_char_count"] == len(TEXT)
    assert debug["attachment_ocr"][0]["ocr_provider"] == "fake-ocr-cpu"
    assert result["answer_provenance"]["research_generation_calls"] == 0
    assert provider.calls == 2  # OCR happens at upload, never reruns during chat.
    store.delete_attachment("owner-123", cid, first["id"])
    with pytest.raises(ValueError, match="Tài liệu"):
        _execute_chat(store, agent, rag, "owner-123", payload, "removed", "central")


def test_no_attachment_keeps_text_contract_and_never_exposes_document_tool(setup):
    _, cid, _, provider, _, _ = setup
    with pytest.raises(ValidationError):
        ChatRequest(conversation_id=cid, question=" ")
    runtime = FakeCentralRuntime([CentralGeneration(content="Thông tin trong tài liệu. [h]")])
    document_tool = FakeTool("search_uploaded_documents", [])
    agent = build_agent(runtime, FakeTool("search_history", [{"chunk_id": "h", "text": TEXT}]), document_tool, has_documents=lambda *args: False)
    result = agent.chat("Tư liệu này nói gì?", owner_id="owner-123", conversation_id=cid)
    assert "search_uploaded_documents" not in result["central_debug"]["allowed_tools"]
    assert document_tool.calls == [] and provider.calls == 0
    assert result["central_debug"]["attachment_count"] == 0 and not result["central_debug"]["ocr_used"]
