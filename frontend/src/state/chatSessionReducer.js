import {
  ANSWER_FAILURE_MESSAGE,
  ANSWER_STOPPED_MESSAGE,
  EVIDENCE_CONTRACT_FAILURE_MESSAGE,
} from "../config/messages.js";
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

function patchMessage(state, messageId, patch) {
  return state.messages.map((message) => {
    if (message.id !== messageId) return message;
    return typeof patch === "function" ? patch(message) : { ...message, ...patch };
  });
}

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

    case "MESSAGES_APPENDED":
      return {
        ...state,
        messages: [...state.messages, ...action.messages],
        status: "processing",
        error: "",
        streamFailed: false,
      };

    case "STREAM_STATUS":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, { status: action.status, mode: action.mode }),
        status: action.status,
      };

    case "STREAM_DELTA":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({
          ...message,
          content: message.content + action.delta,
          status: "streaming",
        })),
        status: "streaming",
      };

    case "STREAM_SOURCES":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, { sources: action.sources }),
        sources: action.sources,
      };

    case "STREAM_DEBUG":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, { debug_trace: action.trace }),
      };

    case "STREAM_ERROR": {
      const fallback = action.kind === "evidence_contract_error"
        ? EVIDENCE_CONTRACT_FAILURE_MESSAGE
        : ANSWER_FAILURE_MESSAGE;

      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({
          ...message,
          content: message.content || fallback,
          status: "error",
          debug_trace: action.trace ?? message.debug_trace,
        })),
        status: "error",
        error: action.message,
        streamFailed: true,
      };
    }

    case "STREAM_DONE": {
      const status = state.streamFailed ? "error" : "done";
      return { ...state, messages: patchMessage(state, action.messageId, { status }), status };
    }

    case "STREAM_ABORTED":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({
          ...message,
          content: message.content || ANSWER_STOPPED_MESSAGE,
          status: "cancelled",
        })),
        status: "cancelled",
      };

    case "STREAM_SYNCED":
      return {
        ...state,
        conversations: action.conversations,
        messages: action.detail.messages,
        attachments: action.detail.attachments,
        sources: getLatestSources(action.detail.messages),
      };

    case "CONVERSATIONS_SET":
      return { ...state, conversations: action.conversations };

    case "SOURCES_SHOWN":
      return { ...state, sources: action.sources };

    case "UPLOAD_QUEUED":
      return { ...state, pendingUploads: [...state.pendingUploads, ...action.items], error: "" };

    case "UPLOAD_PROGRESS":
      return {
        ...state,
        pendingUploads: state.pendingUploads.map((item) =>
          item.id === action.id ? { ...item, status: action.status } : item),
      };

    case "UPLOAD_SETTLED": {
      const pendingUploads = state.pendingUploads.filter((item) => item.id !== action.id);
      if (!action.attachment) {
        return { ...state, pendingUploads, error: action.error ?? state.error };
      }
      return {
        ...state,
        pendingUploads,
        attachments: [
          ...state.attachments.filter((item) => item.id !== action.attachment.id),
          action.attachment,
        ],
      };
    }

    case "ATTACHMENT_REMOVED":
      return {
        ...state,
        attachments: state.attachments.filter((item) => item.id !== action.attachmentId),
      };

    case "ATTACHMENTS_SYNCED":
      return { ...state, conversations: action.conversations, attachments: action.attachments };

    case "ERROR_SET":
      return { ...state, error: action.message };

    default:
      return state;
  }
}
