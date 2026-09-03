import { useRef } from "react";

import {
  BACKEND_UNREACHABLE_MESSAGE,
  CONVERSATION_CREATE_FAILURE_MESSAGE,
  STREAM_FAILURE_MESSAGE,
} from "../config/messages.js";
import { createLocalId } from "../state/ids.js";
import { getSources, normalizeConversationDetail, normalizeConversationList } from "../state/normalizers.js";
import { getConversation, listConversations, streamChat } from "../services/api.js";

export function useChatStream({ dispatch, isRunning, mode, showDebugTrace, ensureActiveConversation }) {
  const abortControllerRef = useRef(null);

  const stop = () => abortControllerRef.current?.abort();

  const submit = async (question) => {
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isRunning) return;

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

    const assistantMessageId = createLocalId("assistant");
    const createdAt = new Date().toISOString();

    dispatch({
      type: "MESSAGES_APPENDED",
      messages: [
        {
          id: createLocalId("user"),
          role: "user",
          content: trimmedQuestion,
          sources: [],
          status: "done",
          created_at: createdAt,
        },
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          sources: [],
          status: "processing",
          created_at: createdAt,
        },
      ],
    });

    const controller = new AbortController();
    abortControllerRef.current = controller;
    let streamFailed = false;

    try {
      await streamChat({
        conversationId,
        question: trimmedQuestion,
        mode,
        finalK: 6,
        debug: showDebugTrace,
        signal: controller.signal,
        onEvent: ({ event: eventName, data }) => {
          if (eventName === "status") {
            dispatch({
              type: "STREAM_STATUS",
              messageId: assistantMessageId,
              status: typeof data === "string" ? data : data?.stage ?? "processing",
              mode: data?.mode ?? mode,
            });
            return;
          }

          if (eventName === "answer_delta") {
            dispatch({
              type: "STREAM_DELTA",
              messageId: assistantMessageId,
              delta: typeof data === "string" ? data : data?.delta ?? "",
            });
            return;
          }

          if (eventName === "sources") {
            dispatch({
              type: "STREAM_SOURCES",
              messageId: assistantMessageId,
              sources: getSources(data),
            });
            return;
          }

          if (eventName === "debug_trace" || eventName === "debug") {
            dispatch({ type: "STREAM_DEBUG", messageId: assistantMessageId, trace: data });
            return;
          }

          if (eventName === "error") {
            streamFailed = true;
            dispatch({
              type: "STREAM_ERROR",
              messageId: assistantMessageId,
              message: typeof data === "string" ? data : data?.message ?? STREAM_FAILURE_MESSAGE,
              kind: data?.type,
              trace: data?.debug_trace,
            });
            return;
          }

          if (eventName === "done") {
            dispatch({ type: "STREAM_DONE", messageId: assistantMessageId });
          }
        },
      });

      if (streamFailed) return;

      const [conversationPayload, detailPayload] = await Promise.all([
        listConversations(),
        getConversation(conversationId),
      ]);

      dispatch({
        type: "STREAM_SYNCED",
        conversations: normalizeConversationList(conversationPayload),
        detail: normalizeConversationDetail(detailPayload),
      });
    } catch (requestError) {
      if (requestError.name === "AbortError") {
        dispatch({ type: "STREAM_ABORTED", messageId: assistantMessageId });
      } else {
        console.error(requestError);
        dispatch({
          type: "STREAM_ERROR",
          messageId: assistantMessageId,
          message: requestError.message || BACKEND_UNREACHABLE_MESSAGE,
        });
      }
    } finally {
      abortControllerRef.current = null;
    }
  };

  return { submit, stop };
}
