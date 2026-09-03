import { useEffect, useLayoutEffect, useRef } from "react";

import {
  ATTACHMENT_DELETE_FAILURE_MESSAGE,
  CONVERSATION_CREATE_FAILURE_MESSAGE,
} from "../config/messages.js";
import { normalizeConversationDetail, normalizeConversationList } from "../state/normalizers.js";
import { createUploadItems, validateUploadSelection } from "../state/uploadQueue.js";
import { deleteAttachment, getConversation, listConversations, uploadAttachment } from "../services/api.js";

export function useAttachments({ dispatch, activeConversationId, attachments = [], isRunning, ensureActiveConversation }) {
  const requestRef = useRef(null);
  const previewsRef = useRef(new Map());
  const mountedRef = useRef(false);
  const conversationRef = useRef(activeConversationId);

  const releasePreview = (id) => {
    const url = previewsRef.current.get(id);
    if (url) URL.revokeObjectURL(url);
    previewsRef.current.delete(id);
  };

  useLayoutEffect(() => {
    const previous = conversationRef.current;
    conversationRef.current = activeConversationId;
    if (previous === activeConversationId || !previous) return;
    const request = requestRef.current;
    if (request) {
      request.abandoned = true;
      request.controller.abort();
      requestRef.current = null;
    }
    for (const url of previewsRef.current.values()) URL.revokeObjectURL(url);
    previewsRef.current.clear();
  }, [activeConversationId]);

  useEffect(() => {
    mountedRef.current = true;
    const previews = previewsRef.current;
    return () => {
      mountedRef.current = false;
      if (requestRef.current) {
        requestRef.current.abandoned = true;
        requestRef.current.controller.abort();
        requestRef.current = null;
      }
      for (const url of previews.values()) URL.revokeObjectURL(url);
      previews.clear();
    };
  }, []);

  const upload = async (selectedFiles, { uploadOrigin = "file" } = {}) => {
    if (isRunning || requestRef.current) {
      dispatch({ type: "ERROR_SET", message: "Vui lòng chờ thao tác hiện tại hoàn tất trước khi thêm ảnh." });
      return;
    }
    const { error, files } = validateUploadSelection(selectedFiles, attachments.length);
    if (error) {
      dispatch({ type: "ERROR_SET", message: error });
      return;
    }
    if (!files.length) return;

    const items = createUploadItems(files).map((item) => {
      const preview_url = item.type.startsWith("image/") ? URL.createObjectURL(item.file) : undefined;
      if (preview_url) previewsRef.current.set(item.id, preview_url);
      return { ...item, preview_url };
    });
    const request = { items, cancelled: new Set(), abandoned: false, controller: new AbortController() };
    requestRef.current = request;
    const isCurrent = () => mountedRef.current && !request.abandoned && requestRef.current === request;
    const sendAction = (action) => {
      if (isCurrent()) dispatch({ ...action, scopeId: request.conversationId });
    };
    dispatch({ type: "UPLOAD_QUEUED", items });

    try {
      request.conversationId = await ensureActiveConversation();
      if (!isCurrent()) return;
      for (const item of items) {
        if (!isCurrent() || request.cancelled.has(item.id)) continue;
        sendAction({ type: "UPLOAD_PROGRESS", id: item.id, status: "processing" });
        try {
          // Finish removed uploads so their eventual server IDs can also be deleted.
          const payload = await uploadAttachment(request.conversationId, item.file, { uploadOrigin });
          const attachment = payload?.attachment ?? payload;
          if (!isCurrent() || request.cancelled.has(item.id)) {
            await deleteAttachment(request.conversationId, attachment.id);
            continue;
          }
          if (item.preview_url) {
            previewsRef.current.delete(item.id);
            previewsRef.current.set(attachment.id, item.preview_url);
          }
          sendAction({ type: "UPLOAD_SETTLED", id: item.id, attachment: { ...attachment, preview_url: item.preview_url } });
          if (attachment.status === "failed") {
            sendAction({ type: "ERROR_SET", message: attachment.error || "Không thể đọc ảnh. Hãy thử ảnh rõ hơn." });
          }
        } catch (requestError) {
          sendAction({ type: "UPLOAD_SETTLED", id: item.id, error: requestError.message || `Không thể xử lý ${item.name}.` });
        } finally {
          releasePreview(item.id);
          sendAction({ type: "UPLOAD_SETTLED", id: item.id });
        }
      }

      if (!isCurrent()) return;
      // Failed OCR records must also remain visible and removable after a refresh.
      try {
        const [conversationPayload, detailPayload] = await Promise.all([
          listConversations({ signal: request.controller.signal }),
          getConversation(request.conversationId, { signal: request.controller.signal }),
        ]);
        sendAction({ type: "ATTACHMENTS_SYNCED", conversations: normalizeConversationList(conversationPayload),
          attachments: normalizeConversationDetail(detailPayload).attachments });
      } catch (refreshError) {
        if (isCurrent()) console.warn("Could not refresh attachments", refreshError);
      }
    } catch (requestError) {
      sendAction({ type: "ERROR_SET", message: requestError.message || CONVERSATION_CREATE_FAILURE_MESSAGE });
    } finally {
      for (const item of items) releasePreview(item.id);
      sendAction({ type: "UPLOAD_FINISHED" });
      if (requestRef.current === request) requestRef.current = null;
    }
  };

  const remove = async (attachmentId) => {
    if (isRunning) return;
    const request = requestRef.current;
    if (request?.items.some((item) => item.id === attachmentId)) {
      request.cancelled.add(attachmentId);
      releasePreview(attachmentId);
      dispatch({ type: "UPLOAD_SETTLED", id: attachmentId, scopeId: request.conversationId });
      return;
    }
    if (!activeConversationId) return;
    const sendAction = (action) => {
      if (mountedRef.current) dispatch({ ...action, scopeId: activeConversationId });
    };
    try {
      await deleteAttachment(activeConversationId, attachmentId);
      releasePreview(attachmentId);
      sendAction({ type: "ATTACHMENT_REMOVED", attachmentId });
      const payload = await listConversations();
      sendAction({ type: "CONVERSATIONS_SET", conversations: normalizeConversationList(payload) });
    } catch (requestError) {
      sendAction({ type: "ERROR_SET", message: requestError.message || ATTACHMENT_DELETE_FAILURE_MESSAGE });
    }
  };

  return { upload, remove, isBusy: () => requestRef.current !== null };
}
