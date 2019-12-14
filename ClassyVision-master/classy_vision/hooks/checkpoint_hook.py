#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import tempfile
from shutil import copy2, move
from typing import Any, Collection, Dict, Optional

from classy_vision import tasks
from classy_vision.generic.distributed_util import is_master
from classy_vision.generic.util import get_checkpoint_dict, save_checkpoint
from classy_vision.hooks.classy_hook import ClassyHook

# 保存模型的钩子函数
class CheckpointHook(ClassyHook):
    """
    Hook to checkpoint a model's task.

    Saves the checkpoints in checkpoint_folder.
    """
    # 这些都是不用实现的
    on_rendezvous = ClassyHook._noop
    on_phase_start = ClassyHook._noop
    on_sample = ClassyHook._noop
    on_forward = ClassyHook._noop
    on_loss_and_meter = ClassyHook._noop
    on_backward = ClassyHook._noop
    on_update = ClassyHook._noop
    on_end = ClassyHook._noop

    def __init__(
        self,
        checkpoint_folder: str,
        input_args: Any,
        phase_types: Optional[Collection[str]] = None,
        checkpoint_period: int = 1,
    ) -> None:
        """The constructor method of CheckpointHook.

        Args:
            checkpoint_folder: Folder to store checkpoints in
            # 保存的参数或生成模型的config文件
            input_args: Any arguments to save about the runtime setup. For example,
                it is useful to store the config that was used to instantiate the model.
            phase_types: If ``phase_types`` is specified, only checkpoint on those phase
                types. Each item in ``phase_types`` must be either "train" or "test".
            checkpoint_period: Checkpoint at the end of every x phases (default 1)

        """
        super().__init__()
        self.checkpoint_folder: str = checkpoint_folder
        self.input_args: Any = input_args
        if phase_types is None:
            phase_types = ["train", "test"]
        assert len(phase_types) > 0 and all(
            phase_type in ["train", "test"] for phase_type in phase_types
        ), "phase_types should contain one or more of ['train', 'test']"
        assert (
            isinstance(checkpoint_period, int) and checkpoint_period > 0
        ), "checkpoint period must be positive"

        self.phase_types: Collection[str] = phase_types
        self.checkpoint_period: int = checkpoint_period
        self.phase_counter: int = 0

    def _save_checkpoint(self, task, filename):
        if getattr(task, "test_only", False):
            return
        assert os.path.exists(
            self.checkpoint_folder
        ), "Checkpoint folder '{}' deleted unexpectedly".format(self.checkpoint_folder)
        # 保存 checkpoint
        # save checkpoint:
        logging.info("Saving checkpoint to '{}'...".format(self.checkpoint_folder))
        checkpoint_file = save_checkpoint(
            self.checkpoint_folder, get_checkpoint_dict(task, self.input_args)
        )
        # 保存最新的checkpoint文件
        # make copy of checkpoint that won't be overwritten:
        if checkpoint_file:
            tmp_dir = tempfile.mkdtemp()
            tmp_file = os.path.join(tmp_dir, filename)
            copy2(checkpoint_file, tmp_file)
            move(tmp_file, os.path.join(self.checkpoint_folder, filename))
    # 训练前要完成的工作
    def on_start(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        # 如果测试阶段，放弃保存模型
        if getattr(task, "test_only", False):
            return
        # 如果没有checkpoint_folder提醒
        if not os.path.exists(self.checkpoint_folder):
            err_msg = "Checkpoint folder '{}' does not exist.".format(
                self.checkpoint_folder
            )
            raise FileNotFoundError(err_msg)

    # phase为一个bath训练完，每一个epoch结束后，保存模型
    def on_phase_end(
        self, task: "tasks.ClassyTask", local_variables: Dict[str, Any]
    ) -> None:
        """Checkpoint the task every checkpoint_period phases.

        We do not necessarily checkpoint the task at the end of every phase.
        """
        # 如果不是master机器，或者phase_type未知，直接返回
        if not is_master() or task.phase_type not in self.phase_types:
            return

        # 没保存一次自增一，循环一次epoch保存模型
        self.phase_counter += 1
        if self.phase_counter % self.checkpoint_period != 0:
            return
        # 重命名模型名
        checkpoint_name = "model_phase-{phase}_end.torch".format(phase=task.phase_idx)
        self._save_checkpoint(task, checkpoint_name)
