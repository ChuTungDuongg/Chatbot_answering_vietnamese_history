import {
  ANSWER_FAILURE_MESSAGE,
  ANSWER_STOPPED_MESSAGE,
} from "../config/messages.js";
import { getLatestSources } from "./normalizers.js";

export const ACTIVE_STATUSES = new Set([
  "request_preparation", "query_analysis", "embedding", "dense_search", "bm25_search", "fusion", "rerank", "context_selection", "prompt_preparation", "tool_selection", "attachment_search", "generation",
  "processing", "retrieval_started", "reranking", "generating", "validating", "validated", "streaming",
  "hybrid_retrieval", "hybrid_answering",
  "central_loading", "central_analyzing", "central_tools", "central_answering",
]);

export function isRunningStatus(status) {
  return ACTIVE_STATUSES.has(status) || status?.startsWith("tool:");
}

function updatePipeline(pipeline = [], event) {
  if (!event?.stage || !event?.state) return pipeline;
  const entry = { stage: event.stage, state: event.state, label: event.message,
    latencyMs: event.latency_ms, requestId: event.request_id,
    provider: event.provider, server: event.server, tool: event.tool };
  const index = pipeline.findIndex((item) => item.stage === event.stage);
  return index < 0 ? [...pipeline, entry] : pipeline.map((item, i) => i === index ? entry : item);
}

function finishPipeline(pipeline = [], state) {
  return pipeline.map((item) => item.state === "started" ? { ...item, state } : item);
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
  streamRunning: false,
  isUploading: false,
  isLoadingConversations: true,
  isLoadingConversation: false,
  isCreatingConversation: false,
};

const CLEARED_THREAD = {
  messages: [], attachments: [], pendingUploads: [], sources: [],
  status: "idle", streamFailed: false, streamRunning: false, isUploading: false,
};

function preservePreviews(current, incoming) {
  return incoming.map((item) => {
    const preview_url = current.find((old) => old.id === item.id)?.preview_url;
    return preview_url ? { ...item, preview_url } : item;
  });
}

function patchMessage(state, messageId, patch) {
  return state.messages.map((message) => {
    if (message.id !== messageId) return message;
    return typeof patch === "function" ? patch(message) : { ...message, ...patch };
  });
}

export function chatSessionReducer(state, action) {
  // Async work is allowed to update only the conversation it started in.
  if (action.scopeId !== undefined && action.scopeId !== state.activeConversationId) return state;
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
      return { ...state, isLoadingConversation: true, sources: [], error: "" };

    case "CONVERSATION_CREATING":
      return { ...state, isCreatingConversation: true, isLoadingConversation: false };

    case "CONVERSATION_CREATE_FINISHED":
      return { ...state, isCreatingConversation: false, isLoadingConversations: false };

    case "CONVERSATION_LOADED":
      return {
        ...state,
        ...CLEARED_THREAD,
        activeConversationId: action.conversationId,
        messages: action.detail.messages,
        attachments: action.detail.attachments,
        sources: getLatestSources(action.detail.messages),
        status: "idle",
        streamFailed: false,
        isLoadingConversation: false,
      };

    case "CONVERSATION_LOAD_FAILED":
      return { ...state, isLoadingConversation: false, sources: getLatestSources(state.messages), error: action.message };

    case "CONVERSATION_CREATED":
      return {
        ...state,
        ...CLEARED_THREAD,
        conversations: [
          action.conversation,
          ...state.conversations.filter((item) => item.id !== action.conversation.id),
        ],
        activeConversationId: action.conversation.id,
        isLoadingConversation: false,
        isLoadingConversations: false,
        // First-conversation creation can be part of an already reserved send/upload.
        streamRunning: !state.activeConversationId && state.streamRunning,
        isUploading: !state.activeConversationId && state.isUploading,
        pendingUploads: !state.activeConversationId ? state.pendingUploads : [],
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
        sources: [],
        status: "processing",
        error: "",
        streamFailed: false,
      };

    case "STREAM_STARTED":
      return { ...state, streamRunning: true, status: "processing", error: "", sources: [], streamFailed: false };

    case "STREAM_FINISHED":
      return { ...state, streamRunning: false, status: action.status ?? state.status };

    case "STREAM_STATUS":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({ ...message,
          status: action.status, mode: action.mode,
          retrieval_backend: action.progress?.retrieval_backend ?? message.retrieval_backend,
          pipeline: updatePipeline(message.pipeline, action.progress) })),
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
      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({
          ...message,
          content: message.content || ANSWER_FAILURE_MESSAGE,
          status: "error",
          pipeline: finishPipeline(message.pipeline, "failed"),
          debug_trace: action.trace ?? message.debug_trace,
        })),
        status: "error",
        error: action.message,
        streamFailed: true,
      };
    }

    case "STREAM_DONE": {
      const status = state.streamFailed ? "error" : "done";
      return { ...state, messages: patchMessage(state, action.messageId, (message) => ({ ...message, status,
        server_id: action.metadata?.message_id ?? message.server_id,
        retrieval_backend: action.metadata?.retrieval_backend ?? message.retrieval_backend,
        pipeline: finishPipeline(message.pipeline, state.streamFailed ? "failed" : "completed") })), status };
    }

    case "STREAM_ABORTED":
      return {
        ...state,
        messages: patchMessage(state, action.messageId, (message) => ({
          ...message,
          content: message.content || ANSWER_STOPPED_MESSAGE,
          status: "cancelled",
          pipeline: finishPipeline(message.pipeline, "cancelled"),
        })),
        status: "cancelled",
      };

    case "STREAM_SYNCED":
      return {
        ...state,
        conversations: action.conversations,
        messages: action.detail.messages.map((message) => {
          const local = state.messages.find((item) => item.server_id && item.server_id === message.id);
          return local ? { ...message, pipeline: local.pipeline, retrieval_backend: local.retrieval_backend } : message;
        }),
        attachments: preservePreviews(state.attachments, action.detail.attachments),
        sources: getLatestSources(action.detail.messages),
      };

    case "CONVERSATIONS_SET":
      return { ...state, conversations: action.conversations };

    case "SOURCES_SHOWN":
      return { ...state, sources: action.sources };

    case "UPLOAD_QUEUED":
      return { ...state, isUploading: true, pendingUploads: [...state.pendingUploads, ...action.items], error: "" };

    case "UPLOAD_FINISHED":
      return { ...state, isUploading: false, pendingUploads: [] };

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
      return { ...state, conversations: action.conversations, attachments: preservePreviews(state.attachments, action.attachments) };

    case "ERROR_SET":
      return { ...state, error: action.message };

    default:
      return state;
  }
}
