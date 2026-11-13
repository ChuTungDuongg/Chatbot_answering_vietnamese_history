import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource-variable/inter";
import "./index.css";
import App from "./App.jsx";
import EvalStudio from "./eval/EvalStudio.jsx";

createRoot(document.getElementById("root")).render(
  <StrictMode>
    {window.location.pathname.startsWith("/eval-studio") ? <EvalStudio /> : <App />}
  </StrictMode>,
);
