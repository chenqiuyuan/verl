# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
#
# ============================================================================
# 按策略决定每条 rollout 的 thinking / non-thinking 模板分支
# ============================================================================
# 背景: Qwen3.5 的 chat template 对 thinking 是"强制"的 ——
#   enable_thinking=False -> '<think>\n\n</think>\n\n'  (强制不思考)
#   否则                  -> '<think>\n'                (强制思考)
# verl 只支持 data.apply_chat_template_kwargs 这一个全局值，于是同一个 run
# 里所有样本被一刀切。而训练数据可能逐回合混合 thinking/non-thinking
# (实测:有的场景约 28% 回合带 <think>,有的约 96%),全局一刀切会把大量
# 回合按与原始数据相反的模板训练,形成训练/数据错配。
#
# 每条样本可读取 extra_info.enable_thinking(由数据管道按原始数据里该回合
# 有无 <think> 写入)覆盖全局默认，实现训推一致。
#
# 决策面: thinking_policy = global | per_sample | per_scene
#   global     verl 原生行为(整个 run 一个值,默认)
#   per_sample 样本 extra_info.enable_thinking → 场景映射 → 全局默认
#   per_scene  按 data_source 查场景映射 → 全局默认(样本标志被忽略)
# 控制键经 data.apply_chat_template_kwargs 传入(+ 前缀新增),__init__ 摘出,
# 不进入模板渲染参数。例如:
#   +data.apply_chat_template_kwargs.thinking_policy=per_scene
#   +data.apply_chat_template_kwargs.scene_thinking_modes.use_skill=false
#   +data.apply_chat_template_kwargs.scene_thinking_modes.reflection=true
#
# 为什么 teacher 不需要改: teacher 只对 student 已生成的 token 串算 logprob
# (agent_loop_tq.py 传的是 prompt_ids / response_ids),不做 chat template、
# 不自己生成，因此自动跟随同一 prompt —— 两边天然一致。
#
# 并发安全: agent_loop.py 的 _run_agent_loop 对每个样本 hydra.instantiate 出
# 独立的 agent loop 实例，因此这里"重新绑定实例属性"不会跨样本串扰;并且用
# dict(...) 拷贝而不是原地改，避免污染共享的 data_config。
# ============================================================================
import logging
import os
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# 只在第一次缺字段时告警，避免每条样本刷屏。
_WARNED_MISSING_FLAG = False

# thinking 模式决策面:
#   thinking_policy = "global"     verl 原生行为,整个 run 一个 enable_thinking(默认)
#   thinking_policy = "per_sample" 样本 extra_info.enable_thinking → 场景映射 → 全局默认
#   thinking_policy = "per_scene"  场景映射(data_source) → 全局默认,样本标志被忽略
# 两个控制键经 data.apply_chat_template_kwargs 传入(+ 前缀新增),在 __init__
# 摘出,不会流入模板渲染参数。默认 global 保持 verl 原生零变更;需要按数据
# 控制 thinking 的训练侧显式传 per_sample/per_scene。
THINKING_POLICIES = ("global", "per_sample", "per_scene")


def _coerce_bool(value: Any) -> bool:
    """extra_info 经 parquet/pandas 回来可能是 numpy.bool_ / 字符串。"""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"enable_thinking 不是合法布尔值: {value!r}")
    return bool(value)


def _extract_enable_thinking(kwargs: dict) -> bool | None:
    """从样本的 extra_info 取 enable_thinking;没有则返回 None(用全局默认)。"""
    extra_info = kwargs.get("extra_info")
    if extra_info is None:
        return None
    # parquet -> pandas 后是 dict;稳妥起见只接受 mapping。
    if not hasattr(extra_info, "get"):
        return None
    if "enable_thinking" not in extra_info:
        return None
    return _coerce_bool(extra_info["enable_thinking"])


def _parse_scene_modes(raw: Any) -> dict[str, bool]:
    """解析场景 → thinking 映射。

    接受两种形态:
      - mapping(hydra CLI 逐键新增: +...scene_thinking_modes.use_skill=false)
      - 字符串 "use_skill=false,reflection=true"(env/脚本侧拼接更方便)
    """
    if raw is None:
        return {}
    if isinstance(raw, str):
        modes: dict[str, bool] = {}
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"scene_thinking_modes 条目缺少 '=': {part!r}")
            scene, _, value = part.partition("=")
            modes[scene.strip()] = _coerce_bool(value)
        return modes
    if hasattr(raw, "items"):
        return {str(k): _coerce_bool(v) for k, v in raw.items()}
    raise ValueError(f"scene_thinking_modes 无法解析: {raw!r}")


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        # 从 apply_chat_template_kwargs 摘出 thinking 控制键,
        # 避免它们被转发进 chat template 渲染上下文。
        base_kwargs = dict(self.apply_chat_template_kwargs or {})
        policy = str(base_kwargs.pop("thinking_policy", "global"))
        if policy not in THINKING_POLICIES:
            raise ValueError(
                f"thinking_policy 必须是 {'/'.join(THINKING_POLICIES)} 之一: {policy!r}"
            )
        self._thinking_policy = policy
        self._scene_thinking_modes = _parse_scene_modes(
            base_kwargs.pop("scene_thinking_modes", None)
        )
        self.apply_chat_template_kwargs = base_kwargs

    def _apply_thinking_policy(self, kwargs: dict) -> None:
        """按 thinking_policy 决定本样本的 thinking 模板分支。

        决策链(细粒度覆盖粗粒度):
          per_sample: 样本 extra_info → 场景映射 → 全局默认
          per_scene:  场景映射 → 全局默认
          global:     不做任何覆盖(verl 原生行为)
        """
        global _WARNED_MISSING_FLAG

        if self._thinking_policy == "global":
            return

        decided: bool | None = None
        if self._thinking_policy == "per_sample":
            decided = _extract_enable_thinking(kwargs)
            if decided is None and not _WARNED_MISSING_FLAG:
                _WARNED_MISSING_FLAG = True
                logger.warning(
                    "[thinking-policy] thinking_policy=per_sample 但样本缺少 "
                    "extra_info.enable_thinking,回退 场景映射→全局默认。"
                    "若期望按数据控制 thinking,请在数据管道写入 extra_info.enable_thinking。",
                )

        if decided is None and self._scene_thinking_modes:
            data_source = kwargs.get("data_source")
            if data_source is None:
                if not _WARNED_MISSING_FLAG:
                    _WARNED_MISSING_FLAG = True
                    logger.warning(
                        "[thinking-policy] 样本缺少 data_source,场景映射(%d 项)不生效,"
                        "回退全局默认。",
                        len(self._scene_thinking_modes),
                    )
            else:
                decided = self._scene_thinking_modes.get(str(data_source))

        if decided is None:
            return

        # continuous token 路径在 __init__ 就用全局 kwargs 构建了 builder,
        # 绕过 apply_chat_template,按样本覆盖不会生效 —— 显式拒绝而不是静默失效。
        if self.enable_continuous_token:
            raise RuntimeError(
                "[thinking-policy] continuous_token 与按样本 enable_thinking 不兼容: "
                "该路径在初始化时固化了全局 chat template kwargs"
            )

        # 拷贝后重新绑定实例属性:不原地修改共享 data_config,实例是每样本独立的。
        merged = dict(self.apply_chat_template_kwargs or {})
        merged["enable_thinking"] = decided
        self.apply_chat_template_kwargs = merged

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], priority: int = 0, **kwargs) -> AgentLoopOutput:
        # priority may arrive as np.int64 from non_tensor_batch; normalize to Python int.
        priority = int(priority)
        messages = list(kwargs["raw_prompt"])

        # 在构造 prompt 之前按 thinking_policy 决定模板分支。
        self._apply_thinking_policy(kwargs)

        # 1. extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        # 2. apply chat template and tokenize
        use_continuous_token = self.enable_continuous_token and not multi_modal_data
        if use_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(messages)
        else:
            prompt_ids = await self.apply_chat_template(
                messages,
                images=images,
                videos=videos,
                audios=audios,
                mm_processor_kwargs=mm_processor_kwargs,
            )

        # 3. generate sequences
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            request_id = f"det-{priority}" if getattr(self.rollout_config, "full_determinism", False) else uuid4().hex
            output: TokenOutput = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                audio_data=audios,
                video_data=videos,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        if use_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                prompt_ids,
                output.token_ids,
                [],
                [] if output.log_probs else None,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            response_ids = merge_result.token_ids[-len(response_mask) :] if response_mask else []
            prompt_ids = merge_result.token_ids[: len(merge_result.token_ids) - len(response_mask)]
        else:
            response_ids = output.token_ids
            response_mask = [1] * len(output.token_ids)
            response_logprobs = output.log_probs

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            num_turns=2,
            metrics=metrics,
            extra_fields=output.extra_fields,
        )

        # keeping the schema consistent with tool_agent_loop
        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})

        return output
