# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm import LLM, SamplingParams

prompts = [
    "1 3 5 7 9",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
llm = LLM(
    model="/data2/models/deepseek-v2-lite",
    enforce_eager=True,
    additional_config={"role": "attn"},
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"prompt{prompt!r}, generated text: {generated_text!r}")
