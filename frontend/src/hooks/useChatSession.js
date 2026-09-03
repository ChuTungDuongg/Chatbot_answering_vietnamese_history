import { useEffect, useReducer } from "react";

import {
  BACKEND_UNREACHABLE_MESSAGE,
  CONVERSATION_CREATE_NEW_FAILURE_MESSAGE,
  CONVERSATION_LOAD_FAILURE_MESSAGE,
  MISSING_CONVERSATION_ID_MESSAGE,
} from "../config/messages.js";
import {
  chatSessionReducer,
  initialChatSessionState,
  isRunningStatus,
} from "../state/chatSessionReducer.js";
import { normalizeConversationDetail, normalizeConversationList } from "../state/normalizers.js";
import {
  createConversation,
  deleteConversation,
  getConversation,
  listConversations,
  updateConversation,
} from "../services/api.js";

export function useChatSession() {
  const [state, dispatch] = useReducer(chatSessionReducer, initialChatSessionState);
  const isRunning = isRunningStatus(state.status);

  useEffect(() => {
    const controller = new AbortController();

    async function bootstrap() {
      try {
        const payload = await listConversations({ signal: controller.signal });
        const items = normalizeConversationList(payload);

        if (items.length === 0) {
          dispatch({ type: "BOOTSTRAPPED", conversations: items });
          return;
        }

        const detailPayload = await getConversation(items[0].id, { signal: controller.signal });
        dispatch({
          type: "BOOTSTRAPPED",
          conversations: items,
          activeConversationId: items[0].id,
          detail: normalizeConversationDetail(detailPayload),
        });
      } catch (requestError) {
        if (requestError.name === "AbortError") return;
        console.error(requestError);
        dispatch({
          type: "BOOTSTRAP_FAILED",
          message: requestError.message || BACKEND_UNREACHABLE_MESSAGE,
        });
      }
    }

    bootstrap();
    return () => controller.abort();
  }, []);

  // Hook tự chịu trách nhiệm tắt cờ đang tải trong CẢ hai nhánh. Nếu để việc đó
  // cho nơi gọi thì chỉ cần một nơi quên là spinner kẹt vĩnh viễn — mà hàm này
  // được gọi từ hai chỗ khác nhau.
  const loadConversation = async (conversationId) => {
    dispatch({ type: "CONVERSATION_LOADING" });
    try {
      const payload = await getConversation(conversationId);
      dispatch({
        type: "CONVERSATION_LOADED",
        conversationId,
        detail: normalizeConversationDetail(payload),
      });
    } catch (requestError) {
      dispatch({
        type: "CONVERSATION_LOAD_FAILED",
        message: requestError.message || CONVERSATION_LOAD_FAILURE_MESSAGE,
      });
      throw requestError;
    }
  };

  const createNewConversation = async () => {
    if (isRunning) throw new Error(CONVERSATION_CREATE_NEW_FAILURE_MESSAGE);

    const payload = await createConversation({ title: null });
    const conversation = payload?.conversation ?? payload;
    if (!conversation?.id) throw new Error(MISSING_CONVERSATION_ID_MESSAGE);

    dispatch({ type: "CONVERSATION_CREATED", conversation });
    return conversation;
  };

  const renameConversation = async (conversation, title) => {
    const updated = await updateConversation(conversation.id, { title });
    dispatch({ type: "CONVERSATION_RENAMED", conversationId: conversation.id, patch: updated });
  };

  const removeConversation = async (conversationId) => {
    await deleteConversation(conversationId);
    dispatch({ type: "CONVERSATION_DELETED", conversationId });
    return state.conversations
      .filter((item) => item.id !== conversationId)
      .map((item) => item.id);
  };

  const ensureActiveConversation = async () => {
    if (state.activeConversationId) return state.activeConversationId;
    const conversation = await createNewConversation();
    return conversation.id;
  };

  return {
    state,
    dispatch,
    isRunning,
    loadConversation,
    createNewConversation,
    renameConversation,
    removeConversation,
    ensureActiveConversation,
  };
}
