import { useEffect, useState } from "react";
import { persistResponseMode, readResponseMode } from "../config/responseModes.js";

export function useResponseMode() {
  const [responseMode, setResponseMode] = useState(readResponseMode);
  useEffect(() => persistResponseMode(responseMode), [responseMode]);
  return { responseMode, setResponseMode };
}
