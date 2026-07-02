from __future__ import annotations

import pytest

from segtask_v1.config import ConfigError, load_config


@pytest.mark.parametrize("yaml_text, expected", [
    (
        """data:
  patch_mode: z_axis
  patch_size: [64, 128, 128]
  multi_res_scales: [1.0]
  label_values: [0, 1]
  aux_keep_native_d: true
model:
  arch: unet
  backbone: resnet
train:
  save_best_criterion: dice
""",
        "use 'keep_native_view_depth' instead",
    ),
    (
        """data:
  patch_mode: z_axis
  patch_size: [64, 128, 128]
  multi_res_scales: [1.0]
  label_values: [0, 1]
model:
  arch: unet
  backbone: resnet
train:
  save_best_criterion: dice
  save_best_metric: mean_dice
""",
        "auto-derived from 'save_best_criterion'",
    ),
    (
        """data:
  patch_mode: z_axis
  patch_size: [64, 128, 128]
  multi_res_scales: [1.0]
  label_values: [0, 1]
model:
  arch: unet
  backbone: resnet
  unknown_future_key: 1
train:
  save_best_criterion: dice
""",
        "Unknown config key 'unknown_future_key' in ModelConfig.",
    ),
    (
        """data:
  patch_mode: z_axis
  patch_size: [64, 128, 128]
  multi_res_scales: [1.0]
  label_values: [0, 1]
model:
  arch: unet
  backbone: resnet
  use_se: true
train:
  save_best_criterion: dice
""",
        'attention_type: "se"',
    ),
])
def test_deprecated_and_unknown_config_keys_raise_config_error(tmp_path, yaml_text, expected):
    path = tmp_path / "bad.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match=expected):
        load_config(str(path))
