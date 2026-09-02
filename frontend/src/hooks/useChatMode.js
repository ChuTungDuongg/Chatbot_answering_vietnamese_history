import { useEffect, useState } from "react";

import { persistChatMode, readStoredChatMode } from "../config/chatModes.js";

export function useChatMode() {
  const [mode, setMode] = useState(readStoredChatMode);

  useEffect(() => {
    persistChatMode(mode);
  }, [mode]);

  return { mode, setMode };
}
