export function normalizeConversationList(payload) {
  if (Array.isArray(payload)) return payload;
  return payload?.items ?? payload?.conversations ?? [];
}

export function normalizeConversationDetail(payload) {
  const conversation = payload?.conversation ?? payload ?? {};
  return {
    conversation,
    messages: payload?.messages ?? conversation.messages ?? [],
    attachments: payload?.attachments ?? conversation.attachments ?? [],
  };
}

export function getSources(data) {
  if (Array.isArray(data)) return data;
  return data?.items ?? data?.sources ?? data?.final_context ?? data?.retrieval?.final_context ?? [];
}

export function getLatestSources(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "assistant" && message.sources?.length) return message.sources;
  }
  return [];
}
