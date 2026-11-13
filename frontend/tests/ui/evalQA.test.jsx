import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import ComparisonWorkspace from "../../src/eval/ComparisonWorkspace";
import ImportPanel from "../../src/eval/ImportPanel";
import EvidenceInspector from "../../src/eval/EvidenceInspector";
import BlindWorkspace from "../../src/eval/BlindWorkspace";
import ExperimentsWorkspace from "../../src/eval/ExperimentsWorkspace";
import EvalStudio from "../../src/eval/EvalStudio";

const snapshots = [{ version: "qa-v1" }];
const experiments = [
  { id: "exp-a", name: "Alpha", snapshot_id: "qa-v1", model_id: "model-a" },
  { id: "exp-b", name: "Beta", snapshot_id: "qa-v1", model_id: "model-b" },
];
function deferred() { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; }
function response(data, status = 200) { return { ok: status < 400, status, json: async () => data }; }
function serve(handler) {
  vi.stubGlobal("fetch", vi.fn(async (url, options = {}) => {
    const path = String(url).replace(/^.*\/api\/v1\/eval/, "");
    const result = await handler(path, options);
    if (result === undefined) throw new Error(`Unhandled request ${options.method || "GET"} ${path}`);
    return result;
  }));
}
function evidence(answer, chunks = []) {
  return { run: { answer, trace: null }, question: { question: "Snapshot question", reference_answer: "Reference", expected_facts: [], must_cover_points: [] }, reviews: [], trace_status: "not_captured", evidence: chunks };
}
function comparison() {
  return { snapshot_id: "qa-v1", configuration_differences: [], experiments: [], rows: [], pairwise: [], paired_questions: 73 };
}
function upload(input, name, content) {
  const file = new File([typeof content === "string" ? content : "[]"], name, { type: "application/json" });
  Object.defineProperty(file, "text", { value: () => typeof content === "string" ? Promise.resolve(content) : content.promise });
  fireEvent.change(input, { target: { files: [file] } });
}
const validPreview = { count: 1, valid: 1, conflicts: [], errors: [], sample: [{ line: 1, id: "Q1", text: "First import" }], token: "token-first" };

beforeEach(() => {
  window.history.replaceState({}, "", "/eval-studio");
  window.localStorage.setItem("vn-history-theme", "light");
});
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.useRealTimers(); });

test("changing comparison filters invalidates an in-flight result", async () => {
  const pending = deferred();
  serve(path => path === "/experiments" ? response(experiments) : path === "/compare" ? pending.promise : undefined);
  render(<ComparisonWorkspace snapshots={snapshots} />);
  fireEvent.click(await screen.findByLabelText(/Alpha/));
  fireEvent.click(screen.getByLabelText(/Beta/));
  fireEvent.click(screen.getByRole("button", { name: "So sánh", exact: true }));
  fireEvent.change(screen.getByLabelText("Danh mục"), { target: { value: "Factual QA" } });
  await act(async () => pending.resolve(response(comparison())));
  expect(screen.queryByText(/73 câu có review/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "So sánh", exact: true })).toBeEnabled();
});

test("a preview response for the previous file cannot authorize the new file", async () => {
  const pending = deferred();
  serve(path => path === "/batch/preview" ? pending.promise : undefined);
  render(<ImportPanel kind="questions" />);
  const input = screen.getByLabelText("Chọn JSON, JSONL hoặc CSV");
  upload(input, "first.json", '[{"id":"Q1"}]');
  await waitFor(() => expect(screen.getByRole("button", { name: /Kiểm tra trước/ })).toBeEnabled());
  fireEvent.click(screen.getByRole("button", { name: /Kiểm tra trước/ }));
  upload(input, "second.json", '[{"id":"Q2"}]');
  await act(async () => pending.resolve(response(validPreview)));
  expect(screen.queryByRole("button", { name: "Xác nhận nhập toàn bộ" })).not.toBeInTheDocument();
});

test("out-of-order file reads keep the newest selected file content", async () => {
  const firstRead = deferred(); let submitted;
  serve((path, options) => { if (path === "/batch/preview") { submitted = JSON.parse(options.body); return response(validPreview); } });
  render(<ImportPanel kind="questions" />);
  const input = screen.getByLabelText("Chọn JSON, JSONL hoặc CSV");
  upload(input, "first.json", firstRead);
  upload(input, "second.json", '[{"id":"Q2"}]');
  await act(async () => {});
  await act(async () => firstRead.resolve('[{"id":"Q1"}]'));
  fireEvent.click(screen.getByRole("button", { name: /Kiểm tra trước/ }));
  await screen.findByRole("button", { name: "Xác nhận nhập toàn bộ" });
  expect(submitted.content).toBe('[{"id":"Q2"}]');
});

test("an oversized replacement file clears the previously importable content", async () => {
  render(<ImportPanel kind="questions" />);
  const input = screen.getByLabelText("Chọn JSON, JSONL hoặc CSV");
  upload(input, "first.json", "[]");
  await act(async () => {});
  const file = new File([""], "large.json"); Object.defineProperty(file, "size", { value: 2_000_001 });
  fireEvent.change(input, { target: { files: [file] } });
  expect(screen.getByRole("button", { name: /Kiểm tra trước/ })).toBeDisabled();
});

test("changing the inspected run immediately hides the previous run evidence", async () => {
  const pending = deferred();
  serve(path => path === "/runs/old/evidence" ? response(evidence("Old answer")) : path === "/runs/new/evidence" ? pending.promise : undefined);
  const view = render(<EvidenceInspector runId="old" />);
  await screen.findByText("Old answer");
  view.rerender(<EvidenceInspector runId="new" />);
  expect(screen.queryByText("Old answer")).not.toBeInTheDocument();
  await act(async () => pending.resolve(response(evidence("New answer"))));
  expect(screen.getByText("New answer")).toBeInTheDocument();
});

test("a reveal cannot show the previous assignment identity on the next question", async () => {
  const pending = deferred();
  const session = { id: "session", snapshot_id: "qa-v1", assignments: [
    { id: "one", question_id: "Q1", question: "First blind question", answer_a: "One A", answer_b: "One B", reviewed: true },
    { id: "two", question_id: "Q2", question: "Second blind question", answer_a: "Two A", answer_b: "Two B", reviewed: true },
  ] };
  serve(path => path === "/experiments" ? response(experiments) : path === "/blind" ? response([{ id: "session", snapshot_id: "qa-v1" }]) : path === "/blind/session" ? response(session) : path.endsWith("/reveal") ? pending.promise : undefined);
  render(<BlindWorkspace snapshots={snapshots} />);
  fireEvent.click(await screen.findByRole("button", { name: /Phiên session/ }));
  fireEvent.click(await screen.findByRole("button", { name: "Reveal identity" }));
  fireEvent.click(screen.getByRole("button", { name: "Sau", exact: true }));
  await act(async () => pending.resolve(response({ a: { name: "Identity from first" }, b: { name: "Other identity" }, review: { preference: "a" } })));
  expect(screen.getByText("Second blind question")).toBeInTheDocument();
  expect(screen.queryByText(/Identity from first/)).not.toBeInTheDocument();
});

test("switching experiments clears an existing run edit before another save", async () => {
  const run = { id: "run-a", experiment_id: "exp-a", question_id: "Q1", answer: "Alpha output", completion_status: "completed" };
  serve(path => path === "/experiments" ? response(experiments) : path === "/runs" ? response([run]) : path === "/runs/run-a/evidence" ? response(evidence("Alpha output")) : undefined);
  render(<ExperimentsWorkspace snapshots={snapshots} />);
  fireEvent.click(await screen.findByRole("button", { name: /Alpha/ }));
  fireEvent.click(screen.getByRole("button", { name: "Sửa run Q1" }));
  await waitFor(() => expect(screen.getByLabelText("Model output")).toHaveValue("Alpha output"));
  fireEvent.click(screen.getByRole("button", { name: /Beta/ }));
  expect(screen.getByLabelText("Model output")).toHaveValue("");
  expect(screen.queryByRole("button", { name: "Cập nhật run" })).not.toBeInTheDocument();
});

const question = id => ({ id, question: `${id} content`, category: "Factual QA", difficulty: "easy", status: "draft", smoke: false, tags: [], version: 1, updated_at: "2026-11-13T00:00:00Z", reference_answer: "", expected_facts: [], must_cover_points: [], reference_sources: [], ranking_criteria: [], uncertainty_notes: "", rubric_overrides: {} });
function studioServer(extra = () => undefined) {
  serve((path, options) => extra(path, options) ?? (path === "/questions" ? response([question("Q1"), question("Q2")]) : path === "/validation" ? response({ warnings: {}, category_counts: {}, quota: {}, duplicates: [], review_metadata_missing: [], reviews_missing_reviewer: [] }) : ["/responses", "/reviews", "/snapshots", "/experiments", "/runs", "/blind"].includes(path) ? response([]) : undefined));
}

test("workspace navigation follows browser Back and Forward history", async () => {
  studioServer(); render(<EvalStudio />);
  await screen.findByRole("heading", { name: "Tổng quan dataset" });
  fireEvent.click(screen.getByRole("button", { name: /^Ngân hàng câu hỏi/ }));
  await screen.findByRole("heading", { name: "Ngân hàng câu hỏi" });
  fireEvent.click(screen.getByRole("button", { name: "Experiments", exact: true }));
  await screen.findByRole("heading", { name: "Experiments" });
  act(() => window.history.back());
  await waitFor(() => expect(screen.getByRole("heading", { name: "Ngân hàng câu hỏi" })).toBeInTheDocument());
  act(() => window.history.forward());
  await screen.findByRole("heading", { name: "Experiments" });
});

test("switching reviewed outputs clears scores that belong to the previous answer", async () => {
  studioServer(path => path === "/responses" ? response([
    { id: "response-a", question_id: "Q1", model: "Model A", answer: "Answer A" },
    { id: "response-b", question_id: "Q1", model: "Model B", answer: "Answer B" },
  ]) : undefined);
  render(<EvalStudio />);
  await screen.findByRole("heading", { name: "Tổng quan dataset" });
  fireEvent.click(screen.getByRole("button", { name: "Human review", exact: true }));
  await screen.findByRole("heading", { name: "Human review" });
  fireEvent.change(screen.getByLabelText("Chọn câu hỏi review"), { target: { value: "Q1" } });
  await waitFor(() => expect(screen.getByLabelText("Chọn câu hỏi review")).toHaveValue("Q1"));
  fireEvent.change(screen.getByLabelText("Output đã lưu"), { target: { value: "response-a" } });
  fireEvent.change(screen.getByLabelText("Điểm Historical Accuracy"), { target: { value: "5" } });
  fireEvent.change(screen.getByLabelText("Output đã lưu"), { target: { value: "response-b" } });
  expect(screen.getByLabelText("Điểm Historical Accuracy")).toHaveValue("");
});

test("v11 Human Review uses immutable run reference instead of the edited draft", async () => {
  const snapshotEvidence = { ...evidence("Run output"), question: { ...question("Q1"), question: "Original snapshot question", reference_answer: "Original snapshot reference" } };
  studioServer(path => path === "/responses" ? response([{ id: "response-a", question_id: "Q1", model: "Model A", answer: "Run output" }]) : path === "/runs" ? response([{ id: "run-a", response_id: "response-a" }]) : path === "/runs/run-a/evidence" ? response(snapshotEvidence) : undefined);
  render(<EvalStudio />);
  await screen.findByRole("heading", { name: "Tổng quan dataset" });
  fireEvent.click(screen.getByRole("button", { name: "Human review", exact: true }));
  await screen.findByRole("heading", { name: "Human review" });
  fireEvent.change(screen.getByLabelText("Chọn câu hỏi review"), { target: { value: "Q1" } });
  await waitFor(() => expect(screen.getByLabelText("Chọn câu hỏi review")).toHaveValue("Q1"));
  fireEvent.change(screen.getByLabelText("Output đã lưu"), { target: { value: "response-a" } });
  await waitFor(() => expect(screen.getAllByText("Original snapshot reference").length).toBeGreaterThan(0));
  expect(screen.queryByText("Q1 content", { selector: "p" })).not.toBeInTheDocument();
});

test("captured empty context and citation lists are distinct from absent trace", async () => {
  const captured = evidence("Output"); captured.run.trace = { final_context_chunk_ids: [], cited_chunk_ids: [], tools: [] }; captured.trace_status = "captured";
  serve(path => path === "/runs/empty/evidence" ? response(captured) : undefined);
  render(<EvidenceInspector runId="empty" />);
  await screen.findByText("Output");
  expect(screen.getByText("Final context: Không có chunk")).toBeInTheDocument();
  expect(screen.getByText("Cited chunks: Không có chunk")).toBeInTheDocument();
  expect(screen.getByText("Tool calls: Không có")).toBeInTheDocument();
});

test("autosave drains edits made during an in-flight save before question navigation", async () => {
  const pending = deferred(), writes = [];
  studioServer((path, options) => {
    if (path === "/questions/Q1" && options.method === "PATCH") {
      const body = JSON.parse(options.body); writes.push(body);
      return writes.length === 1 ? pending.promise : response({ ...body, version: 3 });
    }
  });
  render(<EvalStudio />);
  await screen.findByRole("heading", { name: "Tổng quan dataset" });
  fireEvent.click(screen.getByRole("button", { name: /^Ngân hàng câu hỏi/ }));
  await screen.findByRole("heading", { name: "Ngân hàng câu hỏi" });
  fireEvent.click(await screen.findByRole("button", { name: /Q1.*content/ }));
  const input = await screen.findByRole("textbox", { name: "Câu hỏi", exact: true });
  fireEvent.change(input, { target: { value: "First edit" } });
  fireEvent.keyDown(window, { key: "s", ctrlKey: true });
  await waitFor(() => expect(writes).toHaveLength(1));
  fireEvent.change(input, { target: { value: "Final edit" } });
  fireEvent.click(screen.getByRole("button", { name: "Câu sau" }));
  await act(async () => pending.resolve(response({ ...writes[0], version: 2 })));
  await waitFor(() => expect(input).toHaveValue("Q2 content"));
  expect(writes[1].question).toBe("Final edit");
  expect(writes[1].version).toBe(2);
});

test("a revision conflict preserves the draft, blocks navigation, and guards refresh", async () => {
  studioServer((path, options) => path === "/questions/Q1" && options.method === "PATCH" ? response({ detail: "Revision conflict: reload the latest question" }, 409) : undefined);
  render(<EvalStudio />);
  await screen.findByRole("heading", { name: "Tổng quan dataset" });
  fireEvent.click(screen.getByRole("button", { name: /^Ngân hàng câu hỏi/ }));
  await screen.findByRole("heading", { name: "Ngân hàng câu hỏi" });
  fireEvent.click(await screen.findByRole("button", { name: /Q1.*content/ }));
  const input = await screen.findByRole("textbox", { name: "Câu hỏi", exact: true });
  fireEvent.change(input, { target: { value: "Unsaved local work" } });
  fireEvent.click(screen.getByRole("button", { name: "Experiments", exact: true }));
  await screen.findByRole("alert");
  expect(input).toHaveValue("Unsaved local work");
  expect(screen.queryByRole("heading", { name: "Experiments" })).not.toBeInTheDocument();
  const unload = new Event("beforeunload", { cancelable: true });
  window.dispatchEvent(unload);
  expect(unload.defaultPrevented).toBe(true);
});

test("comparison drawer moves keyboard focus inside and returns it when closed", async () => {
  const data = comparison();
  data.experiments = experiments.map(e => ({ ...e, criteria: {}, metadata_missing: [], diagnosis_counts: {}, distribution: [0, 0, 0, 0, 1] }));
  data.rows = [{ question: question("Q1"), cells: { "exp-a": { run_id: "run-a", review: { weighted_score: 80 } } } }];
  serve(path => path === "/experiments" ? response(experiments) : path === "/compare" ? response(data) : path === "/runs/run-a/evidence" ? response(evidence("Inspected answer")) : undefined);
  render(<ComparisonWorkspace snapshots={snapshots} />);
  fireEvent.click(await screen.findByLabelText(/Alpha/)); fireEvent.click(screen.getByLabelText(/Beta/));
  fireEvent.click(screen.getByRole("button", { name: "So sánh", exact: true }));
  const trigger = await screen.findByRole("button", { name: "80/100" }); trigger.focus(); fireEvent.click(trigger);
  const close = await screen.findByRole("button", { name: "Đóng" });
  expect(close).toHaveFocus();
  fireEvent.keyDown(close, { key: "Tab", shiftKey: true }); expect(close).toHaveFocus();
  fireEvent.keyDown(close, { key: "Escape" });
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(trigger).toHaveFocus();
});

test("experiment creation disables repeated submission while the request is pending", async () => {
  const pending = deferred(); let creations = 0;
  serve((path, options) => path === "/experiments" ? options.method === "POST" ? (creations += 1, pending.promise) : response([]) : path === "/runs" ? response([]) : undefined);
  render(<ExperimentsWorkspace snapshots={snapshots} />);
  fireEvent.click(screen.getByRole("button", { name: "Tạo experiment" }));
  fireEvent.change(screen.getByLabelText("Tên experiment"), { target: { value: "Pending experiment" } });
  fireEvent.change(screen.getByLabelText("Dataset snapshot"), { target: { value: "qa-v1" } });
  const submit = screen.getByRole("button", { name: "Lưu experiment" });
  fireEvent.click(submit); fireEvent.click(submit);
  expect(submit).toBeDisabled();
  expect(creations).toBe(1);
  await act(async () => pending.resolve(response({ id: "new-experiment" })));
});

test("a delayed output save cannot clear the next question response form", async () => {
  const pending = deferred();
  studioServer((path, options) => path === "/responses" && options.method === "POST" ? pending.promise : undefined);
  render(<EvalStudio />);
  await screen.findByRole("heading", { name: "Tổng quan dataset" });
  fireEvent.click(screen.getByRole("button", { name: "Human review", exact: true }));
  await screen.findByRole("heading", { name: "Human review" });
  fireEvent.change(screen.getByLabelText("Chọn câu hỏi review"), { target: { value: "Q1" } });
  await waitFor(() => expect(screen.getByLabelText("Chọn câu hỏi review")).toHaveValue("Q1"));
  fireEvent.change(screen.getByLabelText("Câu trả lời model"), { target: { value: "First output" } });
  fireEvent.click(screen.getByRole("button", { name: "Lưu output" }));
  fireEvent.change(screen.getByLabelText("Chọn câu hỏi review"), { target: { value: "Q2" } });
  await waitFor(() => expect(screen.getByLabelText("Chọn câu hỏi review")).toHaveValue("Q2"));
  fireEvent.change(screen.getByLabelText("Câu trả lời model"), { target: { value: "Next question draft" } });
  await act(async () => pending.resolve(response({ id: "response-first", question_id: "Q1" })));
  expect(screen.getByLabelText("Câu trả lời model")).toHaveValue("Next question draft");
  expect(screen.getByLabelText("Output đã lưu")).toHaveValue("");
});

test("owner-session API and export requests stay on the current Studio origin", async () => {
  vi.stubEnv("VITE_API_BASE_URL", "https://remote-inference.example");
  vi.resetModules();
  const { api, exportUrl, reviewExportUrl } = await import("../../src/eval/api");
  let requested;
  vi.stubGlobal("fetch", async url => { requested = url; return response([]); });
  await api("/questions");
  expect(requested).toBe("/api/v1/eval/questions");
  expect(exportUrl("json", false, false, false)).toBe("/api/v1/eval/export?format=json&smoke=false&approved=false&base_only=false");
  expect(reviewExportUrl).toBe("/api/v1/eval/export/reviews");
});
