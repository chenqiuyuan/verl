# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""single_turn_agent_loop 的 thinking 决策面单测(纯 CPU,无 ray/torch 依赖)。

决策链(细粒度覆盖粗粒度):
  per_sample: 样本 extra_info.enable_thinking → 场景映射 → 全局默认
  per_scene:  场景映射(data_source) → 全局默认
  global:     verl 原生行为,不做任何覆盖(默认)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from verl.experimental.agent_loop.agent_loop import AgentLoopBase
from verl.experimental.agent_loop.single_turn_agent_loop import (
    SingleTurnAgentLoop,
    _coerce_bool,
    _extract_enable_thinking,
    _parse_scene_modes,
)


def _make_loop(policy: str, scene_modes=None, apply_kwargs=None) -> SingleTurnAgentLoop:
    """绕过 AgentLoopBase.__init__ 构造最小实例,只装配决策面所需属性。"""
    loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
    loop.prompt_length = 8
    loop.response_length = 8
    loop.apply_chat_template_kwargs = dict(apply_kwargs or {})
    loop._thinking_policy = policy
    loop._scene_thinking_modes = _parse_scene_modes(scene_modes)
    loop.enable_continuous_token = False
    return loop


def _init_via_patch(apply_kwargs: dict) -> SingleTurnAgentLoop:
    """mock 掉 AgentLoopBase.__init__ 后走真实 __init__,验证控制键摘取。"""
    with mock.patch.object(AgentLoopBase, "__init__", return_value=None):
        loop = SingleTurnAgentLoop.__new__(SingleTurnAgentLoop)
        loop.rollout_config = SimpleNamespace(prompt_length=8, response_length=8)
        loop.apply_chat_template_kwargs = dict(apply_kwargs)
        SingleTurnAgentLoop.__init__(loop)
    return loop


# ---------------------------------------------------------------- 校验函数
def test_coerce_bool_accepts_strings_and_bools():
    assert _coerce_bool("true") is True
    assert _coerce_bool("0") is False
    assert _coerce_bool(True) is True
    with pytest.raises(ValueError):
        _coerce_bool("maybe")


def test_extract_enable_thinking_missing_returns_none():
    assert _extract_enable_thinking({}) is None
    assert _extract_enable_thinking({"extra_info": None}) is None
    assert _extract_enable_thinking({"extra_info": {}}) is None
    assert _extract_enable_thinking({"extra_info": {"enable_thinking": False}}) is False


def test_parse_scene_modes_string_and_mapping():
    assert _parse_scene_modes("use_skill=false,reflection=true") == {
        "use_skill": False,
        "reflection": True,
    }
    assert _parse_scene_modes({"use_skill": "true"}) == {"use_skill": True}
    assert _parse_scene_modes(None) == {}
    with pytest.raises(ValueError):
        _parse_scene_modes("use_skill=false,broken")


# ---------------------------------------------------------------- __init__ 摘取
def test_init_pops_control_keys_and_defaults_global():
    loop = _init_via_patch(
        {"thinking_policy": "per_scene", "scene_thinking_modes": {"use_skill": "false"}, "other": 1}
    )
    assert loop._thinking_policy == "per_scene"
    assert loop._scene_thinking_modes == {"use_skill": False}
    # 控制键不能流入模板渲染参数
    assert loop.apply_chat_template_kwargs == {"other": 1}


def test_init_default_policy_is_global():
    loop = _init_via_patch({})
    assert loop._thinking_policy == "global"


def test_init_rejects_invalid_policy():
    with pytest.raises(ValueError, match="thinking_policy"):
        _init_via_patch({"thinking_policy": "per_universe"})


# ---------------------------------------------------------------- 决策链
def test_global_policy_never_overrides():
    loop = _make_loop("global", apply_kwargs={"enable_thinking": True})
    loop._apply_thinking_policy({"extra_info": {"enable_thinking": False}, "data_source": "use_skill"})
    assert loop.apply_chat_template_kwargs == {"enable_thinking": True}


def test_per_sample_three_level_fallback():
    # 1) 样本标志最优先
    loop = _make_loop(
        "per_sample", scene_modes="use_skill=false", apply_kwargs={"enable_thinking": True}
    )
    loop._apply_thinking_policy({"extra_info": {"enable_thinking": False}, "data_source": "use_skill"})
    assert loop.apply_chat_template_kwargs["enable_thinking"] is False
    # 2) 无样本标志 → 场景映射
    loop._apply_thinking_policy({"data_source": "use_skill"})
    assert loop.apply_chat_template_kwargs["enable_thinking"] is False
    # 3) 都没有 → 全局默认(不覆盖)
    loop._apply_thinking_policy({"data_source": "unknown_scene"})
    assert loop.apply_chat_template_kwargs["enable_thinking"] is True


def test_per_scene_ignores_sample_flag():
    loop = _make_loop(
        "per_scene", scene_modes="use_skill=false", apply_kwargs={"enable_thinking": True}
    )
    # 样本标志为 True 也被忽略,场景映射说了算
    loop._apply_thinking_policy({"extra_info": {"enable_thinking": True}, "data_source": "use_skill"})
    assert loop.apply_chat_template_kwargs["enable_thinking"] is False
    # 未知场景 → 全局默认
    loop._apply_thinking_policy({"data_source": "unknown"})
    assert loop.apply_chat_template_kwargs["enable_thinking"] is True


def test_continuous_token_incompatible_raises():
    loop = _make_loop(
        "per_sample", scene_modes="use_skill=false", apply_kwargs={"enable_thinking": True}
    )
    loop.enable_continuous_token = True
    with pytest.raises(RuntimeError, match="continuous_token"):
        loop._apply_thinking_policy({"data_source": "use_skill"})
