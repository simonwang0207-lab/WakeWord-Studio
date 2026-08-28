"""Shared deterministic flags for the DS-TC-ResNet smoke POC."""

from __future__ import annotations

import copy
from pathlib import Path

from kws_streaming.models import model_flags, model_params


def make_flags(data_dir: Path, train_dir: Path):
    flags = copy.deepcopy(model_params.HOTWORD_MODEL_PARAMS["ds_tc_resnet"])
    flags.data_url = ""
    flags.data_dir = str(data_dir)
    flags.train_dir = str(train_dir)
    flags.train = 1
    flags.restore_checkpoint = 0
    flags.split_data = 0
    flags.wanted_words = "qingxiaojia,other"
    flags.clip_duration_ms = 1000
    flags.window_size_ms = 30.0
    flags.window_stride_ms = 10.0
    flags.mel_num_bins = 40
    flags.dct_num_features = 20
    flags.feature_type = "mfcc_tf"
    flags.use_tf_fft = 1
    flags.mel_non_zero_only = 0
    flags.batch_size = 4
    flags.how_many_training_steps = "2"
    flags.learning_rate = "0.001"
    flags.eval_step_interval = 1
    flags.pick_deterministically = 1
    flags.resample = 0.0
    flags.time_shift_ms = 50.0
    flags.background_frequency = 0.8
    flags.background_volume = 0.1
    flags.return_softmax = 0
    flags.use_spec_augment = 0
    flags.ds_padding = "'causal','causal','causal','causal'"
    flags.ds_filters = "16,16,16,16"
    flags.ds_repeat = "1,1,1,1"
    flags.ds_residual = "0,1,1,0"
    flags.ds_kernel_size = "5,7,9,1"
    flags.ds_stride = "1,1,1,1"
    flags.ds_dilation = "1,1,2,1"
    flags.ds_pool = "1,1,1,1"
    flags.ds_filter_separable = "1,1,1,1"
    return model_flags.update_flags(flags)
