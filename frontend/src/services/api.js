const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function parseSSEBlock(block) {
  let event = "message";
  const dataLines = [];

  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (!dataLines.length) {
    return null;
  }

  const rawData = dataLines.join("\n");
  let data = rawData;

  try {
    data = JSON.parse(rawData);
  } catch {
    // Keep plain-text SSE payload as a string.
  }

  return { event, data };
}

export async function streamChat({
  question,
  finalK = 6,
  debug = true,
  onEvent,
  signal,
}) {
  if (!API_BASE_URL) {
    throw new Error("VITE_API_BASE_URL is not configured.");
  }

  const response = await fetch(`${API_BASE_URL}/api/v1/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({
      question,
      final_k: finalK,
      debug,
    }),
    signal,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}.`);
  }

  if (!response.body) {
    throw new Error("Streaming response body is unavailable.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });

    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const parsed = parseSSEBlock(block);

      if (parsed) {
        onEvent?.(parsed);
      }
    }
  }

  buffer += decoder.decode();

  if (buffer.trim()) {
    const parsed = parseSSEBlock(buffer);

    if (parsed) {
      onEvent?.(parsed);
    }
  }
}