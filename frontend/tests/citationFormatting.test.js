import assert from "node:assert/strict";
import test from "node:test";
import { displayAnswer, remarkSourceCitations } from "../src/services/citations.js";

test("legacy source brackets are display-only and numeric years remain untouched", () => {
  const sources = [{ source_id: "hf_wikipedia_history_001", display_index: 2 }];
  const original = structuredClone(sources);
  assert.equal(displayAnswer("Dữ kiện [hf_wikipedia_history_001], năm [1954].", sources), "Dữ kiện [2], năm [1954].");
  assert.deepEqual(sources, original);
});

test("only known display indices become citation links", () => {
  const tree = { type: "paragraph", children: [{ type: "text", value: "Năm [938], [1945], [1954]; nguồn [1], [2], [9]." }] };
  remarkSourceCitations({ sources: [{ display_index: 1 }, { display_index: 2 }] })(tree);
  assert.deepEqual(tree.children.filter((node) => node.type === "link").map((node) => node.url), ["#source-1", "#source-2"]);
  assert.ok(tree.children.some((node) => node.value?.includes("[1954]")));
  assert.ok(tree.children.some((node) => node.value?.includes("[9]")));
});

test("citation formatting leaves code and existing links intact", () => {
  const tree = { type: "root", children: [
    { type: "code", value: "[1]" },
    { type: "inlineCode", value: "[1]" },
    { type: "link", url: "https://example.org", children: [{ type: "text", value: "[1]" }] },
  ] };
  const original = structuredClone(tree);
  remarkSourceCitations({ sources: [{ display_index: 1 }] })(tree);
  assert.deepEqual(tree, original);
});
