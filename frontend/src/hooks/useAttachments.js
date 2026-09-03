import {
  ATTACHMENT_DELETE_FAILURE_MESSAGE,
  CONVERSATION_CREATE_FAILURE_MESSAGE,
} from "../config/messages.js";
import { normalizeConversationDetail, normalizeConversationList } from "../state/normalizers.js";
import { createUploadItems, validateUploadSelection } from "../state/uploadQueue.js";
import {
  deleteAttachment,
  getConversation,
  listConversations,
  uploadAttachment,
} from "../services/api.js";

export function useAttachments({ dispatch, activeConversationId, isRunning, ensureActiveConversation }) {
  const upload = async (selectedFiles) => {
    if (isRunning) return;

    const { error, files } = validateUploadSelection(selectedFiles);
    if (error) {
      dispatch({ type: "ERROR_SET", message: error });
      return;
    }

    let conversationId;
    try {
      conversationId = await ensureActiveConversation();
    } catch (requestError) {
      dispatch({
        type: "ERROR_SET",
        message: requestError.message || CONVERSATION_CREATE_FAILURE_MESSAGE,
      });
      return;
    }

    const items = createUploadItems(files);
    dispatch({ type: "UPLOAD_QUEUED", items });

    for (const item of items) {
      dispatch({ type: "UPLOAD_PROGRESS", id: item.id, status: "processing" });

      try {
        const payload = await uploadAttachment(conversationId, item.file);
        dispatch({
          type: "UPLOAD_SETTLED",
          id: item.id,
          attachment: payload?.attachment ?? payload,
        });
      } catch (requestError) {
        console.error(requestError);
        dispatch({
          type: "UPLOAD_SETTLED",
          id: item.id,
          error: requestError.message || `Không thể xử lý ${item.name}.`,
        });
      }
    }

    try {
      const [conversationPayload, detailPayload] = await Promise.all([
        listConversations(),
        getConversation(conversationId),
      ]);

      dispatch({
        type: "ATTACHMENTS_SYNCED",
        conversations: normalizeConversationList(conversationPayload),
        attachments: normalizeConversationDetail(detailPayload).attachments,
      });
    } catch (refreshError) {
      console.warn("Could not refresh attachments", refreshError);
    }
  };

  const remove = async (attachmentId) => {
    if (!activeConversationId || isRunning) return;

    try {
      await deleteAttachment(activeConversationId, attachmentId);
      dispatch({ type: "ATTACHMENT_REMOVED", attachmentId });

      const payload = await listConversations();
      dispatch({ type: "CONVERSATIONS_SET", conversations: normalizeConversationList(payload) });
    } catch (requestError) {
      console.error(requestError);
      dispatch({
        type: "ERROR_SET",
        message: requestError.message || ATTACHMENT_DELETE_FAILURE_MESSAGE,
      });
    }
  };

  return { upload, remove };
}
