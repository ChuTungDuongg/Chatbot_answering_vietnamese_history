import { useEffect, useLayoutEffect, useReducer, useRef } from "react";

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
  const isRunning = state.streamRunning || isRunningStatus(state.status);
  const requestRef = useRef(null);
  const creationRef = useRef(null);
  const mountedRef = useRef(false);
  // Read the latest committed session after an await, rather than the caller's render.
  const stateRef = useRef(state);
  useLayoutEffect(() => { stateRef.current = state; }, [state]);

  useEffect(() => {
    const controller = new AbortController();
    mountedRef.current = true;
    requestRef.current = controller;

    async function bootstrap() {
      try {
        const payload = await listConversations({ signal: controller.signal });
        if (controller.signal.aborted) return;
        const items = normalizeConversationList(payload);
        dispatch({ type: "CONVERSATIONS_SET", conversations: items });

        if (items.length === 0) {
          dispatch({ type: "BOOTSTRAPPED", conversations: items });
          return;
        }

        const detailPayload = await getConversation(items[0].id, { signal: controller.signal });
        if (controller.signal.aborted) return;
        dispatch({
          type: "BOOTSTRAPPED",
          conversations: items,
          activeConversationId: items[0].id,
          detail: normalizeConversationDetail(detailPayload),
        });
      } catch (requestError) {
        if (controller.signal.aborted || requestError.name === "AbortError") return;
        console.error(requestError);
        dispatch({
          type: "BOOTSTRAP_FAILED",
          message: requestError.message || BACKEND_UNREACHABLE_MESSAGE,
        });
      }
    }

    bootstrap();
    return () => {
      mountedRef.current = false;
      controller.abort();
      requestRef.current?.abort();
    };
  }, []);

  // Hook tự chịu trách nhiệm tắt cờ đang tải trong CẢ hai nhánh. Nếu để việc đó
  // cho nơi gọi thì chỉ cần một nơi quên là spinner kẹt vĩnh viễn — mà hàm này
  // được gọi từ hai chỗ khác nhau.
  const loadConversation = async (conversationId) => {
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    dispatch({ type: "CONVERSATION_LOADING" });
    try {
      const payload = await getConversation(conversationId, { signal: controller.signal });
      if (controller.signal.aborted) return false;
      dispatch({
        type: "CONVERSATION_LOADED",
        conversationId,
        detail: normalizeConversationDetail(payload),
      });
      return true;
    } catch (requestError) {
      if (controller.signal.aborted) return false;
      dispatch({
        type: "CONVERSATION_LOAD_FAILED",
        message: requestError.message || CONVERSATION_LOAD_FAILURE_MESSAGE,
      });
      throw requestError;
    }
  };

  const createNewConversation = async () => {
    if (creationRef.current) return creationRef.current;
    if (isRunning) throw new Error(CONVERSATION_CREATE_NEW_FAILURE_MESSAGE);
    dispatch({ type: "CONVERSATION_CREATING" });

    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    const pending = (async () => {
      const payload = await createConversation({ title: null, signal: controller.signal });
      if (controller.signal.aborted) throw new DOMException("Aborted", "AbortError");
      const conversation = payload?.conversation ?? payload;
      if (!conversation?.id) throw new Error(MISSING_CONVERSATION_ID_MESSAGE);
      dispatch({ type: "CONVERSATION_CREATED", conversation });
      return conversation;
    })();
    creationRef.current = pending;
    try {
      return await pending;
    } finally {
      if (creationRef.current === pending) creationRef.current = null;
      if (mountedRef.current) dispatch({ type: "CONVERSATION_CREATE_FINISHED" });
    }
  };

  const renameConversation = async (conversation, title) => {
    const updated = await updateConversation(conversation.id, { title });
    if (!mountedRef.current) return;
    dispatch({ type: "CONVERSATION_RENAMED", conversationId: conversation.id, patch: updated });
  };

  const removeConversation = async (conversationId) => {
    await deleteConversation(conversationId);
    if (!mountedRef.current) return [];
    dispatch({ type: "CONVERSATION_DELETED", conversationId });
    return stateRef.current.conversations
      .filter((item) => item.id !== conversationId)
      .map((item) => item.id);
  };

  const ensureActiveConversation = async () => {
    if (stateRef.current.activeConversationId) return stateRef.current.activeConversationId;
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
