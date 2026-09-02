# Central V2 repository refactor and offline tooling

The current host state was preserved, then reorganized by ownership. Canonical
locations and commands are documented in [agent packages](../app/agents/README.md),
[V2 training](../training/central/README.md), and [future evaluation](../evaluation/README.md).

## Runtime audit

- Research, Evidence, History Answerer and Central each have one canonical agent
  class in their own package. Legacy orchestration lives in `three_llm/` and Hybrid
  keeps its existing orchestrator in `hybrid.py`.
- Shared runtime loading, model registry/cache, tool codec, comparison utilities,
  domain gate and the common evidence chunk belong to `common/`. Role-specific
  configs, schemas and prompts stay with their owner.
- The four root `config.py`, `schemas.py`, `prompts.py`, and `orchestrator.py` files
  are documented compatibility re-exports only. No duplicate business logic remains.
- AST comparison against the pre-refactor working state found no changed Central
  function/class bodies after excluding import statements. The three legacy agent
  classes, both orchestrators and both configuration classes were also preserved.
- Central semantic values/normalization were moved below question analysis.
  Evidence packet planning moved out of depth policy, and paragraph support moved
  below citation validation. The dependency test checks even deferred imports for
  cycles, along with forbidden cross-package dependencies.
- No new historical proper-name dispatch or question-specific production rule was
  added. Existing lexical/entity conventions were moved unchanged; this audit does
  not claim that the pre-existing code contains no historical names.
- Central does not import/delegate to legacy agents or fall back to `three_llm`.
  Production does not import training/evaluation; common does not import concrete
  agents; training/evaluation do not import each other's internals.
- Backend/API/scripts/tests use canonical imports. Modal helper edits only migrate
  imports; deployment/scaling settings and artifact lifecycle are unchanged. OCR,
  clipboard, attachments and frontend behavior were not intentionally changed.

## Training and evaluation preparation

The existing Hermes and ViQuAD normalizers and configurable trainer now have one
canonical implementation under `training/central`. The V2 CLI gates inputs to the
two intended sources before loading tokenizer/model. Tool argument schemas are
validated offline; the centralized seeded mix does not duplicate examples. Native
Qwen tool formatting and assistant-only supervision remain the shared preprocessing
contract. The stable `training.train_qwen3_8b_agent` command delegates to this trainer.

Evaluation has JSON schemas/contracts, two synthetic fixtures, an explicit future
same-host runner, raw logs, seven metric groups, conservative missing-field handling,
strict paired configuration/adapter checks, and offline JSON/Markdown/CSV reporting.
No real benchmark, quality scores, model run, judge run or dataset build was produced.
Generated training data and evaluation logs/reports are ignored.

## Verification

- Full local Python suite: **969 passed, 1 skipped**. The skip is the deliberately
  absent legacy V1 Colab notebook.
- Final focused package/training/evaluation/relational regression rerun: **91 passed**.
- `python -m compileall -q app training evaluation scripts tests modal_app.py`: passed.
- `git diff --check`: passed.
- Candidate-file audit: only intended source/docs/configuration/tests and tiny
  fixtures; no generated logs, reports, model weights, cache, checkpoints or indexes.
- No training, model/GPU inference, Modal command, adapter upload, large model or
  dataset download was executed. Only CPU/static/fake tests ran.

The host is ready to freeze for the next training phase based on local regressions.
Semantic extraction, support and summary classification remain conservative surface
heuristics. Real source curation, tokenizer/model/hardware preflight, V2 training,
artifact handoff and later paired evaluation remain future work. Current citation
traces do not provide complete denominators for every proposed citation metric;
those outputs deliberately remain N/A rather than inventing quality measurements.
