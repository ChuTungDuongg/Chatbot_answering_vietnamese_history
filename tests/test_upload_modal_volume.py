from __future__ import annotations

from argparse import Namespace

from scripts.upload_modal_volume import collect_uploads


def test_bundle_upload_targets_volume_root_without_extra_wrapper(tmp_path):
    bundle = tmp_path / "vn_history_deployment"
    (bundle / "history_answerer").mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    args = Namespace(
        local_dir=str(bundle),
        remote_dir="/",
        history_model=None,
        history_adapter=None,
        research_agent=None,
        evidence_agent=None,
        retrieval_dir=None,
        corpus=None,
        config_dir=None,
        manifest=None,
    )
    uploads = collect_uploads(args)
    assert {upload.remote for upload in uploads} == {"/history_answerer", "/manifest.json"}
