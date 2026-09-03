import { useCallback, useEffect, useRef } from "react";

import {
  BACKEND_UNREACHABLE_MESSAGE,
  CONVERSATION_CREATE_FAILURE_MESSAGE,
  STREAM_FAILURE_MESSAGE,
} from "../config/messages.js";
import { createLocalId } from "../state/ids.js";
import { getSources, normalizeConversationDetail, normalizeConversationList } from "../state/normalizers.js";
import { getConversation, listConversations, streamChat } from "../services/api.js";

export function useChatStream({ dispatch, activeConversationId, isRunning, isUploading,
  attachments = [], mode, showDebugTrace, ensureActiveConversation }) {
  const requestRef = useRef(null);

  const stop = useCallback(() => {
    const request = requestRef.current;
    if (!request) return;
    request.controller.abort();
    requestRef.current = null;
    dispatch({ type: "STREAM_ABORTED", scopeId: request.conversationId, messageId: request.messageId });
    dispatch({ type: "STREAM_FINISHED", scopeId: request.conversationId });
  }, [dispatch]);

  useEffect(() => {
    const request = requestRef.current;
    if (request?.conversationId && request.conversationId !== activeConversationId) stop();
  }, [activeConversationId, stop]);

  useEffect(() => () => {
    requestRef.current?.controller.abort();
    requestRef.current = null;
  }, []);

  const submit = async (question) => {
    const trimmedQuestion = question.trim();
    const readyAttachments = attachments.filter((item) => item.status === "ready");
    if ((!trimmedQuestion && !readyAttachments.length) || isRunning || isUploading || requestRef.current) return false;

    // Reserve synchronously, including the first-conversation await and final refresh.
    const controller = new AbortController();
    const request = { controller, conversationId: activeConversationId };
    requestRef.current = request;
    const isCurrent = () => requestRef.current === request && !controller.signal.aborted;
    dispatch({ type: "STREAM_STARTED" });

    let conversationId;
    try {
      conversationId = await ensureActiveConversation();
      if (!isCurrent()) return false;
      request.conversationId = conversationId;
    } catch (requestError) {
      if (!isCurrent()) return false;
      dispatch({
        type: "ERROR_SET",
        message: requestError.message || CONVERSATION_CREATE_FAILURE_MESSAGE,
      });
      dispatch({ type: "STREAM_FINISHED", status: "error" });
      requestRef.current = null;
      return false;
    }

    const assistantMessageId = createLocalId("assistant");
    request.messageId = assistantMessageId;
    const createdAt = new Date().toISOString();
    const sendAction = (action) => {
      if (isCurrent()) dispatch({ ...action, scopeId: conversationId });
    };

    sendAction({
      type: "MESSAGES_APPENDED",
      messages: [
        {
          id: createLocalId("user"),
          role: "user",
          content: trimmedQuestion,
          sources: readyAttachments.map((item) => ({ chunk_id: `attachment:${item.id}`,
            attachment_id: item.id, title: item.filename, source_kind: "attachment" })),
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

    let streamFailed = false;

    try {
      await streamChat({
        conversationId,
        question: trimmedQuestion,
        attachmentIds: readyAttachments.map((item) => item.id),
        mode,
        finalK: 6,
        debug: showDebugTrace,
        signal: controller.signal,
        onEvent: ({ event: eventName, data }) => {
          if (!isCurrent() || streamFailed) return;
          if (eventName === "status") {
            sendAction({
              type: "STREAM_STATUS",
              messageId: assistantMessageId,
              status: typeof data === "string" ? data : data?.stage ?? "processing",
              mode: data?.mode ?? mode,
            });
            return;
          }

          if (eventName === "answer_delta") {
            sendAction({
              type: "STREAM_DELTA",
              messageId: assistantMessageId,
              delta: typeof data === "string" ? data : data?.delta ?? "",
            });
            return;
          }

          if (eventName === "sources") {
            sendAction({
              type: "STREAM_SOURCES",
              messageId: assistantMessageId,
              sources: getSources(data),
            });
            return;
          }

          if (eventName === "debug_trace" || eventName === "debug") {
            sendAction({ type: "STREAM_DEBUG", messageId: assistantMessageId, trace: data });
            return;
          }

          if (eventName === "error") {
            streamFailed = true;
            sendAction({
              type: "STREAM_ERROR",
              messageId: assistantMessageId,
              message: typeof data === "string" ? data : data?.message ?? STREAM_FAILURE_MESSAGE,
              kind: data?.type,
              trace: data?.debug_trace,
            });
            return;
          }

          if (eventName === "done") {
            sendAction({ type: "STREAM_DONE", messageId: assistantMessageId });
          }
        },
      });

      if (streamFailed || !isCurrent()) return true;
      sendAction({ type: "STREAM_DONE", messageId: assistantMessageId });

      const [conversationPayload, detailPayload] = await Promise.all([
        listConversations({ signal: controller.signal }),
        getConversation(conversationId, { signal: controller.signal }),
      ]);

      sendAction({
        type: "STREAM_SYNCED",
        conversations: normalizeConversationList(conversationPayload),
        detail: normalizeConversationDetail(detailPayload),
      });
    } catch (requestError) {
      if (requestError.name === "AbortError") {
        sendAction({ type: "STREAM_ABORTED", messageId: assistantMessageId });
      } else {
        console.error(requestError);
        sendAction({
          type: "STREAM_ERROR",
          messageId: assistantMessageId,
          message: requestError.message || BACKEND_UNREACHABLE_MESSAGE,
        });
      }
    } finally {
      if (requestRef.current === request) {
        dispatch({ type: "STREAM_FINISHED", scopeId: conversationId });
        requestRef.current = null;
      }
    }
    return true;
  };

  return { submit, stop, isBusy: () => requestRef.current !== null };
}
