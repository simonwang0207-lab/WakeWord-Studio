from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs/models/repcnn_performance_v2_1_robust_finetune.yaml"
RUNNER = PROJECT_ROOT / "phase6/scripts/run_b2_1_robust_finetune.py"


def test_b2_1_plan_is_separate_short_and_test_closed():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["model_name"] == "qingxiaojia_repcnn_performance_v2_1_robust_finetune"
    assert 500 <= config["fine_tune"]["planned_steps"] <= 1000
    assert config["fine_tune"]["planned_steps"] == 750
    assert config["frozen_data_contract"]["allowed_splits"] == ["train", "validation"]
    assert config["frozen_data_contract"]["held_out_test_loaded"] is False
    assert config["frozen_data_contract"]["live_diagnostic_wavs_for_tuning"] is False
    assert config["augmentation"]["temporal_shift"]["maximum_frames"] == 3
    assert config["augmentation"]["mild_spec_augment"]["maximum_time_mask_frames"] == 3
    assert config["augmentation"]["mild_spec_augment"]["maximum_frequency_mask_bins"] == 2
    assert config["augmentation"]["microphone_eq"]["enabled"] is False


def test_b2_1_runner_has_non_mutating_preflight_before_run_directory_creation():
    source = RUNNER.read_text(encoding="utf-8")

    assert 'parser.add_argument("--preflight-only"' in source
    assert '"training_started": False' in source
    assert source.index("if args.preflight_only:") < source.index(
        "run_dir.mkdir(parents=True, exist_ok=True)"
    )
