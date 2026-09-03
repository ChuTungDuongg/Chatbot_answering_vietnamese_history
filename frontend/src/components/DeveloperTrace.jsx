import { useState } from "react";
import { Braces, Check, ChevronDown, Copy } from "lucide-react";
import { formatTraceForCopy, hasTraceSection, sanitizeTraceForCopy } from "../services/trace";

function TraceSection({ label, value, defaultOpen = false, tone = "default" }) {
  if (!hasTraceSection(value)) return null;

  return (
    <details className={`trace-section trace-section-${tone}`} open={defaultOpen || undefined}>
      <summary>{label}</summary>
      <pre>{JSON.stringify(sanitizeTraceForCopy(value), null, 2)}</pre>
    </details>
  );
}

function DeveloperTrace({ trace }) {
  const [copied, setCopied] = useState(false);
  if (!hasTraceSection(trace)) return null;

  const copyTrace = async () => {
    try {
      await navigator.clipboard.writeText(formatTraceForCopy(trace));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1600);
    } catch {
      setCopied(false);
    }
  };

  return (
    <details className="developer-trace">
      <summary>
        <Braces />
        <span>Agent trace</span>
        <ChevronDown className="trace-chevron" />
      </summary>

      <div className="trace-panel-body">
        <div className="trace-panel-toolbar">
          <span>{trace.mode || trace.request?.mode || "Pipeline trace"}</span>
          <button type="button" onClick={copyTrace}>
            {copied ? <Check /> : <Copy />}
            {copied ? "Copied" : "Copy trace"}
          </button>
        </div>

        <TraceSection label="Request" value={trace.request} defaultOpen />
        <TraceSection label="Retrieval" value={trace.retrieval} defaultOpen />
        <TraceSection label="Research" value={trace.research} />
        <TraceSection label="Evidence" value={trace.evidence} />
        <TraceSection label="History" value={trace.history} />
        <TraceSection label="Tools" value={trace.tool_trace} />
        <TraceSection label="Analysis" value={trace.analysis} />
        <TraceSection label="Answer provenance" value={trace.answer_provenance} />
        <TraceSection label="Sources" value={trace.sources} />
        <TraceSection label="Performance" value={trace.performance} />
        <TraceSection label="Errors" value={trace.errors} tone="error" defaultOpen />
      </div>
    </details>
  );
}

export default DeveloperTrace;
