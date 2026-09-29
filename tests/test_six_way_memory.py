"""Memory and resume regressions without downloading a model."""

import asyncio
import gc
import json
import threading
import weakref
from types import SimpleNamespace

import pytest

from app.config import HYBRID_MODEL_ID
from app.models.qwen import QwenRuntime
from evaluation.gpu_memory import CudaMemory, MIB, append_sample, read_samples, summarize


class FakeCuda:
    def __init__(self, available=True):
        self.available = available
        self.reset_count = self.empty_count = 0

    def is_available(self):
        return self.available

    def current_device(self):
        return 0

    def get_device_name(self, index):
        return "fake GPU"

    def memory_allocated(self, index):
        return 3 * MIB

    def memory_reserved(self, index):
        return 5 * MIB

    def max_memory_allocated(self, index):
        return 7 * MIB

    def max_memory_reserved(self, index):
        return 9 * MIB

    def reset_peak_memory_stats(self, index):
        self.reset_count += 1

    def empty_cache(self):
        self.empty_count += 1


def test_cuda_telemetry_and_cpu_guard(tmp_path):
    cuda = FakeCuda()
    monitor = CudaMemory("cuda", SimpleNamespace(cuda=cuda))
    monitor.reset_peaks()
    sample = monitor.sample()
    monitor.empty_cache()
    assert (sample["allocated_mib"], sample["reserved_mib"],
            sample["peak_allocated_mib"], sample["peak_reserved_mib"]) == (3, 5, 7, 9)
    assert sample["device_name"] == "fake GPU"
    assert (cuda.reset_count, cuda.empty_count) == (1, 1)
    path = tmp_path / "gpu_memory.jsonl"
    append_sample(path, {"event": "question", "system": "vanilla_no_rag",
                         "question_index": 1, "before_generation": sample,
                         "after_generation": sample, "after_cleanup": sample})
    append_sample(path, {"event": "question", "system": "vanilla_no_rag",
                         "question_index": 2, "before_generation": sample,
                         "after_generation": sample, "after_cleanup": sample})
    with path.open("ab") as stream:
        stream.write(b'{"event":"unfinished"')
    assert len(read_samples(path)) == 2
    append_sample(path, {"event": "system_teardown", "system": "vanilla_no_rag"})
    assert len(read_samples(path)) == 3
    assert summarize(path, "vanilla_no_rag")["allocated_mib"]["slope_mib_per_question"] == 0
    assert CudaMemory("cpu", SimpleNamespace(cuda=cuda)).sample() is None
    unavailable = FakeCuda(available=False)
    cpu = CudaMemory("cuda", SimpleNamespace(cuda=unavailable))
    cpu.reset_peaks(); cpu.empty_cache()
    assert cpu.sample() is None
    assert (unavailable.reset_count, unavailable.empty_count) == (0, 0)


@pytest.mark.parametrize("raises", [False, True])
def test_qwen_generate_joins_worker_and_releases_inputs(raises):
    import torch
    input_refs = []

    class Batch(dict):
        def to(self, device):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, prompt, **kwargs):
            tensor = torch.tensor([[50]])
            input_refs.append(weakref.ref(tensor))
            return Batch(input_ids=tensor)

        def decode(self, ids, **kwargs):
            return "answer"

    class Model:
        def get_input_embeddings(self):
            return SimpleNamespace(weight=SimpleNamespace(device="cpu"))

        def generate(self, *, input_ids, streamer, stopping_criteria, **kwargs):
            assert torch.is_inference_mode_enabled()
            if raises:
                raise RuntimeError("fake generation failure")
            streamer.put(input_ids)
            streamer.put(torch.tensor([[1]]))
            stopping_criteria(torch.tensor([[50, 1]]), None)
            streamer.end()

    runtime = QwenRuntime(model_id=HYBRID_MODEL_ID)
    runtime.tokenizer = Tokenizer()
    runtime.model = Model()
    before = {thread.ident for thread in threading.enumerate() if thread.name == "qwen-generate"}
    if raises:
        with pytest.raises(RuntimeError, match="fake generation failure"):
            asyncio.run(runtime.generate([{"role": "user", "content": "x"}], max_new_tokens=4))
    else:
        answer, done = asyncio.run(runtime.generate([{"role": "user", "content": "x"}], max_new_tokens=4))
        assert answer == "answer" and done.output_tokens == 1
    assert {thread.ident for thread in threading.enumerate() if thread.name == "qwen-generate"} == before
    assert not runtime._generate_lock.locked()
    gc.collect()
    assert input_refs[0]() is None


def test_resume_preserves_old_prediction_and_identity(tmp_path, monkeypatch):
    import app.models.qwen as qwen
    import evaluation.six_way as six
    from tools.drive_cli import sha256

    test_file = tmp_path / "test.jsonl"
    questions = [{"id": "q1", "question": "First?", "category": "chronology"},
                 {"id": "q2", "question": "Second?", "category": "chronology"}]
    test_file.write_text("".join(json.dumps(q) + "\n" for q in questions), encoding="utf-8")
    monkeypatch.setattr(six, "FROZEN_CANONICAL_SHA", "tiny-test-only")
    monkeypatch.setattr(six, "FROZEN_TEST_SHA", sha256(test_file))
    (tmp_path / "split_manifest.json").write_text(json.dumps({"seed": 42,
        "canonical_dataset_sha256": "tiny-test-only", "split_sha256": {"test": sha256(test_file)},
        "ids": {"test": ["q1", "q2"]}}), encoding="utf-8")
    generated = []

    class Model:
        def __init__(self, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["enable_thinking"] is False
            assert kwargs["dtype"] == "bfloat16"

        async def generate(self, messages, *, max_new_tokens):
            assert max_new_tokens == 768
            generated.append(messages[-1]["content"])
            return "answer", SimpleNamespace(input_tokens=1, output_tokens=1,
                                              model_id="fake", model_revision="fixture")

    monkeypatch.setattr(qwen, "QwenRuntime", Model)
    monkeypatch.setattr(six, "_git_commit", lambda: "old-commit")
    args = six.build_parser().parse_args(["--test-file", str(test_file), "--systems", "vanilla_no_rag",
        "--output-dir", str(tmp_path / "run"), "--corpus-path", str(tmp_path / "corpus.jsonl"),
        "--retrieval-root", str(tmp_path / "retrieval"), "--device", "cpu"])
    asyncio.run(six.run(args))
    predictions_path = args.output_dir / "vanilla_no_rag" / "predictions.jsonl"
    first_line = predictions_path.read_bytes().splitlines(keepends=True)[0]
    predictions_path.write_bytes(first_line)  # Simulate an interruption after q1.
    manifest_path = args.output_dir / "run_manifest.json"
    old_identity = json.loads(manifest_path.read_text(encoding="utf-8"))["identity"]
    monkeypatch.setattr(six, "_git_commit", lambda: "new-commit")
    generated.clear()
    args.resume = True
    asyncio.run(six.run(args))
    assert len(generated) == 1 and "Second?" in generated[0]
    assert predictions_path.read_bytes().startswith(first_line)
    resumed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed_manifest["identity"] == old_identity
    assert resumed_manifest["git_commit"] == "old-commit"
    assert resumed_manifest["resume_git_commits"] == ["new-commit"]
    assert (args.output_dir / "gpu_memory.jsonl").exists()
    assert [s.name for s in six.select_systems("all")] == list(six.SYSTEM_NAMES)
