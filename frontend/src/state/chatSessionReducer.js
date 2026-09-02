import { getLatestSources } from "./normalizers.js";

export const ACTIVE_STATUSES = new Set([
  "processing", "retrieval_started", "reranking", "generating", "validating", "validated", "streaming",
  "hybrid_retrieval", "hybrid_answering",
  "three_llm_research", "three_llm_evidence", "three_llm_answering",
  "central_analyzing", "central_tools", "central_answering",
]);

export function isRunningStatus(status) {
  return ACTIVE_STATUSES.has(status);
}

export const initialChatSessionState = {
  conversations: [],
  activeConversationId: null,
  messages: [],
  attachments: [],
  pendingUploads: [],
  sources: [],
  status: "idle",
  error: "",
  streamFailed: false,
  isLoadingConversations: true,
  isLoadingConversation: false,
};

const CLEARED_THREAD = { messages: [], attachments: [], sources: [] };

export function chatSessionReducer(state, action) {
  switch (action.type) {
    case "BOOTSTRAPPED": {
      const base = { ...state, conversations: action.conversations, isLoadingConversations: false };
      if (!action.detail) return base;
      return {
        ...base,
        activeConversationId: action.activeConversationId ?? null,
        messages: action.detail.messages,
        attachments: action.detail.attachments,
        sources: getLatestSources(action.detail.messages),
      };
    }

    case "BOOTSTRAP_FAILED":
      return { ...state, isLoadingConversations: false, error: action.message };

    case "CONVERSATION_LOADING":
      return { ...state, isLoadingConversation: true, error: "" };

    case "CONVERSATION_LOADED":
      return {
        ...state,
        activeConversationId: action.conversationId,
        messages: action.detail.messages,
        attachments: action.detail.attachments,
        sources: getLatestSources(action.detail.messages),
        status: "idle",
        streamFailed: false,
        isLoadingConversation: false,
      };

    case "CONVERSATION_LOAD_FAILED":
      return { ...state, isLoadingConversation: false, error: action.message };

    case "CONVERSATION_CREATED":
      return {
        ...state,
        ...CLEARED_THREAD,
        conversations: [
          action.conversation,
          ...state.conversations.filter((item) => item.id !== action.conversation.id),
        ],
        activeConversationId: action.conversation.id,
        status: "idle",
        streamFailed: false,
      };

    case "CONVERSATION_RENAMED":
      return {
        ...state,
        conversations: state.conversations.map((item) =>
          item.id === action.conversationId ? { ...item, ...action.patch } : item),
      };

    case "CONVERSATION_DELETED": {
      const conversations = state.conversations.filter((item) => item.id !== action.conversationId);
      if (state.activeConversationId !== action.conversationId) return { ...state, conversations };
      return { ...state, ...CLEARED_THREAD, conversations, activeConversationId: null };
    }

    case "CONVERSATIONS_SET":
      return { ...state, conversations: action.conversations };

    case "SOURCES_SHOWN":
      return { ...state, sources: action.sources };

    case "ERROR_SET":
      return { ...state, error: action.message };

    default:
      return state;
  }
}
