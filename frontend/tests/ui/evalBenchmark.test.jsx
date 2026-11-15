import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import EvalStudio from "../../src/eval/EvalStudio";
import EvidenceInspector from "../../src/eval/EvidenceInspector";
import AutomatedComparison from "../../src/eval/AutomatedComparison";

const experiments = [
  { id: "exp-a", name: "Alpha", snapshot_id: "v12", provider: "openai", model_id: "Luna", chat_mode: "hybrid", dense_backend: "faiss", final_k: 6 },
  { id: "exp-b", name: "Beta", snapshot_id: "v12", provider: "openai", model_id: "Luna", chat_mode: "central", dense_backend: "faiss", final_k: 6 },
];
const capabilities = { live_available: true, reason: null, prompt_resources: ["system"], models: ["Luna"], tools: ["search_local"] };
const question = { id: "Q1", question: "Ai là người lãnh đạo?", category: "Factual QA", difficulty: "easy", status: "draft", smoke: true, tags: [], version: 1, updated_at: "2026-11-16T00:00:00Z", reference_answer: "Tham chiếu", expected_facts: [], must_cover_points: [], reference_sources: [], ranking_criteria: [], acceptable_alternatives: [], common_mistakes: [], uncertainty_notes: "", rubric_overrides: {} };
function job(id = "job-a", status = "ready") { return { id, experiment_id: "exp-a", snapshot_id: "v12", status, configuration_hash: `config-${id}`, prompt_hash: "prompt-hash", tool_hash: "tool-hash", configuration: experiments[0], options: { max_retries: 0 }, counts: { total: 1, pending: status === "ready" ? 1 : 0, running: status === "running" ? 1 : 0, completed: status === "completed" ? 1 : 0, failed: 0, cancelled: 0 }, items: [{ question_id: "Q1", status: status === "completed" ? "completed" : "pending", attempts: 0, run_id: status === "completed" ? "run-a" : null, error_code: null }], lease_expires_at: null }; }
const response = (data, status = 200) => ({ ok: status < 400, status, json: async () => data });
function serve(extra = () => undefined) {
  vi.stubGlobal("fetch", vi.fn(async (url, options = {}) => {
    const path = String(url).replace(/^.*\/api\/v1\/eval/, "");
    const custom = await extra(path, options);
    if (custom !== undefined) return custom;
    if (path === "/questions") return response([question]);
    if (path === "/validation") return response({ warnings: {}, category_counts: {}, quota: {}, duplicates: [], review_metadata_missing: [], reviews_missing_reviewer: [] });
    if (path === "/snapshots") return response([{ version: "v12" }]);
    if (path === "/experiments") return response(experiments);
    if (path === "/benchmark/capabilities") return response(capabilities);
    if (["/responses", "/reviews", "/runs", "/benchmarks"].includes(path)) return response([]);
    throw new Error(`Unhandled ${options.method || "GET"} ${path}`);
  }));
}
beforeEach(() => { window.history.replaceState({}, "", "/eval-studio?workspace=benchmarks"); window.localStorage.setItem("vn-history-theme", "light"); });
afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

test("creating a benchmark freezes a preview but requires separate explicit paid start", async () => {
  let created, starts = 0;
  serve((path, options) => {
    if (path === "/benchmarks" && options.method === "POST") { created = JSON.parse(options.body); return response(job(), 201); }
    if (path === "/benchmarks/job-a/start") { starts++; expect(JSON.parse(options.body)).toEqual({ confirm_live: true }); return response(job("job-a", "running")); }
  });
  render(<EvalStudio />);
  await screen.findByRole("heading", { name: "Benchmarks" });
  fireEvent.change(await screen.findByLabelText("Experiment benchmark"), { target: { value: "exp-a" } });
  fireEvent.change(screen.getByLabelText("Prompt overrides (JSON)"), { target: { value: '{"system":"Chỉ dùng nguồn đã cung cấp."}' } });
  fireEvent.click(screen.getByRole("button", { name: "Xem cấu hình trước khi tạo" }));
  expect(screen.getByLabelText("Xem trước benchmark")).toHaveTextContent("Luna");
  fireEvent.click(screen.getByRole("button", { name: "Tạo benchmark chưa chạy" }));
  await screen.findByText("config-job-a");
  expect(created).toMatchObject({ experiment_id: "exp-a", max_retries: 0, requests_per_minute: 6, prompt_overrides: { system: "Chỉ dùng nguồn đã cung cấp." } });
  expect(starts).toBe(0);
  const start = screen.getByRole("button", { name: "Start · gọi model có tính phí" });
  expect(start).toBeDisabled();
  fireEvent.click(screen.getByLabelText(/Tôi xác nhận gọi model có tính phí/));
  fireEvent.click(start);
  await screen.findByText("Đang chạy", { selector: ".benchmark-state" });
  expect(starts).toBe(1);
});

test("preview is invalidated when benchmark options change and invalid variant keys cannot be created", async () => {
  serve(); render(<EvalStudio />);
  await screen.findByLabelText("Experiment benchmark");
  fireEvent.change(await screen.findByLabelText("Experiment benchmark"), { target: { value: "exp-a" } });
  fireEvent.click(screen.getByRole("button", { name: "Xem cấu hình trước khi tạo" }));
  fireEvent.change(screen.getByLabelText("Số lần retry tối đa"), { target: { value: "1" } });
  expect(screen.queryByRole("button", { name: "Tạo benchmark chưa chạy" })).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Prompt overrides (JSON)"), { target: { value: '{"unknown_resource":"No"}' } });
  fireEvent.click(screen.getByRole("button", { name: "Xem cấu hình trước khi tạo" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("unknown_resource");
  expect(screen.queryByRole("button", { name: "Tạo benchmark chưa chạy" })).not.toBeInTheDocument();
});

test("unavailable live execution explains the limitation and blocks start", async () => {
  serve(path => path === "/benchmark/capabilities" ? response({ ...capabilities, live_available: false, reason: "Provider chưa được cấu hình" }) : path === "/benchmarks" ? response([job()]) : path === "/benchmarks/job-a" ? response(job()) : undefined);
  render(<EvalStudio />);
  fireEvent.click(await screen.findByRole("button", { name: /job-a/ }));
  await screen.findByText("config-job-a");
  expect(screen.getByText("Provider chưa được cấu hình")).toBeInTheDocument();
  fireEvent.click(screen.getByLabelText(/Tôi xác nhận gọi model có tính phí/));
  expect(screen.getByRole("button", { name: "Start · gọi model có tính phí" })).toBeDisabled();
});

test("active benchmark polling stops at terminal state and on unmount", async () => {
  let reads = 0;
  serve(path => path === "/benchmarks" ? response([job("job-a", "running")]) : path === "/benchmarks/job-a" ? response(job("job-a", ++reads > 1 ? "completed" : "running")) : undefined);
  const view = render(<EvalStudio />);
  fireEvent.click(await screen.findByRole("button", { name: /job-a/ }));
  await screen.findByText("config-job-a");
  await waitFor(() => expect(screen.getByText("Hoàn tất", { selector: ".benchmark-state" })).toBeInTheDocument(), { timeout: 4000 });
  vi.useFakeTimers();
  const completedReads = reads;
  await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
  expect(reads).toBe(completedReads);
  view.unmount();
  await act(async () => { await vi.advanceTimersByTimeAsync(10000); });
  expect(reads).toBe(completedReads);
});

test("paused benchmark resume is explicit and cancel displays durable status", async () => {
  let starts = 0;
  serve((path, options) => path === "/benchmarks" ? response([job("job-a", "paused")]) : path === "/benchmarks/job-a" ? response(job("job-a", "paused")) : path.endsWith("/start") ? (starts++, response(job("job-a", "running"))) : path.endsWith("/cancel") && options.method === "POST" ? response(job("job-a", "cancelled")) : undefined);
  render(<EvalStudio />);
  fireEvent.click(await screen.findByRole("button", { name: /job-a/ }));
  await screen.findByText("config-job-a");
  expect(starts).toBe(0);
  const resume = screen.getByRole("button", { name: "Resume · gọi model có tính phí" });
  expect(resume).toBeDisabled();
  fireEvent.click(screen.getByLabelText(/Tôi xác nhận gọi model có tính phí/));
  fireEvent.click(resume);
  await screen.findByText("Đang chạy", { selector: ".benchmark-state" });
  fireEvent.click(screen.getByRole("button", { name: "Hủy benchmark" }));
  expect(await screen.findByText("Đã hủy", { selector: ".benchmark-state" })).toBeInTheDocument();
  expect(starts).toBe(1);
});

test("an expired worker lease offers owner-confirmed resume without restarting automatically", async () => {
  const stale = { ...job("job-a", "running"), lease_expires_at: 1, error_code: "worker_stopped" };
  let starts = 0;
  serve(path => path === "/benchmarks" ? response([stale]) : path === "/benchmarks/job-a" ? response(stale) : path.endsWith("/start") ? (starts++, response(job("job-a", "running"))) : undefined);
  render(<EvalStudio />);
  fireEvent.click(await screen.findByRole("button", { name: /job-a/ }));
  expect(await screen.findByText(/Lease của worker đã hết hạn/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Resume · gọi model có tính phí" })).toBeDisabled();
  expect(starts).toBe(0);
});

test("a cancel acknowledgement still running continues polling until cancellation is durable", async () => {
  let cancelled = false, reads = 0;
  serve((path, options) => path === "/benchmarks" ? response([job("job-a", "running")]) : path.endsWith("/cancel") && options.method === "POST" ? (cancelled = true, response({ ...job("job-a", "running"), cancel_requested: true })) : path === "/benchmarks/job-a" ? (reads++, response(job("job-a", cancelled ? "cancelled" : "running"))) : undefined);
  render(<EvalStudio />);
  fireEvent.click(await screen.findByRole("button", { name: /job-a/ }));
  fireEvent.click(await screen.findByRole("button", { name: "Hủy benchmark" }));
  expect(await screen.findByText("Đang yêu cầu hủy…")).toBeInTheDocument();
  await waitFor(() => expect(screen.getByText("Đã hủy", { selector: ".benchmark-state" })).toBeInTheDocument(), { timeout: 4000 });
  expect(reads).toBeGreaterThan(1);
});

test("measurement inspector renders safe deterministic checks without exposing raw diagnostics", async () => {
  serve(path => path.endsWith("/evidence") ? response({ run: { answer: "Câu trả lời", trace: null }, question, reviews: [], trace_status: "captured", evidence: [] }) : path.endsWith("/measurements") ? response({ metrics: { checks: { required_tools: true }, answer_metrics: { token_f1: 0.7 } }, diagnostics: { tool_calls: [{ name: "search_local", status: "success", arguments_valid: true, argument_fingerprint: "sha256:safe", duration_ms: 12 }], planning: { exit_reason: "answered" }, raw_messages: "PRIVATE REASONING", api_key: "SECRET" } }) : undefined);
  render(<EvidenceInspector runId="run-a" />);
  const checks = await screen.findByRole("region", { name: "Đo lường tự động" });
  expect(checks).toHaveTextContent("required_tools");
  expect(checks).toHaveTextContent("0.700");
  expect(checks).toHaveTextContent("không xác minh tính đúng đắn lịch sử");
  expect(screen.getByText("sha256:safe")).toBeInTheDocument();
  expect(screen.queryByText(/PRIVATE REASONING|SECRET/)).not.toBeInTheDocument();
});

test("legacy run measurements 404 leaves evidence readable", async () => {
  serve(path => path.endsWith("/evidence") ? response({ run: { answer: "Output cũ", trace: null }, question, reviews: [], trace_status: "not_captured", evidence: [] }) : path.endsWith("/measurements") ? response({ detail: "Not found" }, 404) : undefined);
  render(<EvidenceInspector runId="legacy" />);
  expect(await screen.findByText("Output cũ")).toBeInTheDocument();
  expect(await screen.findByText("Chưa ghi đo lường cho run này.")).toBeInTheDocument();
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

test("expected behavior rejects rigid sequences and malformed JSON without autosaving them", async () => {
  let saved;
  window.history.replaceState({}, "", "/eval-studio?workspace=bank");
  serve((path, options) => path === "/questions/Q1" && options.method === "PATCH" ? (saved = JSON.parse(options.body), response({ ...saved, version: 2 })) : undefined);
  render(<EvalStudio />);
  fireEvent.click(await screen.findByRole("button", { name: /Q1.*Ai là người lãnh đạo/ }));
  const editor = await screen.findByLabelText("Expected behavior (JSON)");
  fireEvent.change(editor, { target: { value: '{"tool_sequence":["search_local"]}' } });
  expect(screen.getByRole("alert")).toHaveTextContent("tool_sequence");
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 800)); });
  expect(saved).toBeUndefined();
  fireEvent.change(editor, { target: { value: '{"required_tools":["search_local"],"max_tool_calls":3}' } });
  await waitFor(() => expect(saved?.expected_behavior).toEqual({ required_tools: ["search_local"], max_tool_calls: 3 }));
});

test("automated comparison keeps missing metrics unknown and human scores separate", async () => {
  let request;
  serve((path, options) => path === "/benchmark/compare" ? (request = JSON.parse(options.body), response({ snapshot_id: "v12", comparable: true, configuration_differences: ["chat_mode"], metric_directions: {}, experiments: experiments.map((e, i) => ({ ...e, benchmark_id: `job-${i}`, total_questions: 1, evaluated_questions: i ? 0 : 1, failed_questions: 0, missing_measurements: i ? 1 : 0, metrics: { "answer_metrics.token_f1": i ? null : 0.8 }, latency_ms: null, input_tokens: null, output_tokens: null, tool_calls: null, cost: null, quality: { human_mean: i ? null : 80, judge_mean: null } })), pairwise: [], rows: [] })) : undefined);
  render(<EvalStudio />);
  await screen.findByRole("heading", { name: "Benchmarks" });
  fireEvent.click(screen.getByRole("button", { name: "So sánh tự động" }));
  const panel = screen.getByRole("region", { name: "So sánh đo lường tự động" });
  fireEvent.click(await within(panel).findByLabelText(/Alpha/));
  fireEvent.click(within(panel).getByLabelText(/Beta/));
  fireEvent.click(within(panel).getByRole("button", { name: "So sánh đo lường" }));
  expect(await within(panel).findByText("0.800")).toBeInTheDocument();
  expect(within(panel).getAllByText("Chưa ghi").length).toBeGreaterThan(0);
  expect(within(panel).getByText(/Điểm human/)).toBeInTheDocument();
  expect(request).toEqual({ experiment_ids: ["exp-a", "exp-b"] });
});

test.each([
  ["higher", 0.2, 3, "B cải thiện so với A", "+0.200"],
  ["higher", -0.2, 3, "B hồi quy so với A", "-0.200"],
  ["lower", -0.2, 3, "B cải thiện so với A", "-0.200"],
  ["lower", 0.2, 3, "B hồi quy so với A", "+0.200"],
  ["higher", 0, 3, "Không đổi", "0.000"],
  ["higher", null, 3, "Chưa đủ cặp dữ liệu", "Chưa ghi"],
  ["higher", 0.2, 0, "Chưa đủ cặp dữ liệu", "Chưa ghi"],
  [null, 0.2, 3, "Chưa rõ chiều đánh giá", "+0.200"],
])("paired %s metric with delta %s and %s pairs reports %s", async (direction, delta, paired, label, shownDelta) => {
  const metric = "tools.success_rate";
  serve(path => path === "/benchmark/compare" ? response({ snapshot_id: "v12", comparable: true, configuration_differences: [], metric_directions: direction ? { [metric]: direction } : {}, experiments: experiments.map(exp => ({ ...exp, benchmark_id: "job", total_questions: 3, evaluated_questions: 3, failed_questions: 0, missing_measurements: 0, metrics: {}, latency_ms: null, input_tokens: null, output_tokens: null, tool_calls: null, cost: null, quality: { human_mean: null, judge_mean: null } })), pairwise: [{ a: "exp-a", b: "exp-b", metric, paired, wins: 1, ties: 1, losses: 1, mean_delta: delta }], rows: [] }) : undefined);
  render(<AutomatedComparison experiments={experiments}/>);
  fireEvent.click(screen.getByLabelText(/Alpha/));
  fireEvent.click(screen.getByLabelText(/Beta/));
  fireEvent.click(screen.getByRole("button", { name: "So sánh đo lường" }));
  const table = await screen.findByRole("table", { name: "Chênh lệch A/B theo chỉ số" });
  const row = within(table).getByRole("row", { name: /tools.success_rate/ });
  expect(row).toHaveTextContent("A: Alpha");
  expect(row).toHaveTextContent("B: Beta");
  expect(row).toHaveTextContent(label);
  expect(within(row).getAllByRole("cell")[0]).toHaveTextContent(shownDelta);
  expect(screen.getByText(/Δ = B − A/)).toBeInTheDocument();
  if (label !== "B hồi quy so với A") expect(within(row).queryByText("B hồi quy so với A")).not.toBeInTheDocument();
});
