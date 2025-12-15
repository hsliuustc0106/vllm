from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

import json
from datasets import load_dataset
from task_templates import TASK_TEMPLATES


def predict_on_data(args, task="qasper", sample_size=None):
    llm = LLM(**args)
    dataset = load_dataset("zai-org/LongBench", task, split="test", trust_remote_code=True)
    task_temp = TASK_TEMPLATES.get(task)
    
    conversations = []
    for idx, item in enumerate(dataset):
        conversations.append([
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": task_temp["template"].format(context=item["context"], input=item["input"])},
        ])
        if idx + 1 == sample_size:
            break

    sampling_params = llm.get_default_sampling_params()
    sampling_params.temperature = 0 # set temperature to 0 for greedy decoding
    sampling_params.max_tokens = 128

    outputs = llm.chat(conversations, sampling_params, use_tqdm=True)
    for idx in range(len(outputs)):
        pred_data = {
            "sample_idx": idx,
            "pred": outputs[idx].outputs[0].text,
            "answers": dataset[idx]["answers"],
        }
        with open(f"pred/longbench_{task}_preds.jsonl", "a") as f:
            json.dump(pred_data, f, ensure_ascii=False)
            f.write("\n")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Add engine args
    EngineArgs.add_cli_args(parser)
    # enforce eager execution (no dynamo, no cudagraphs)
    parser.set_defaults(enforce_eager=True)

    parser.set_defaults(model="meta-llama/Llama-3.1-8B-Instruct")
    parser.set_defaults(max_model_len=32768)
    parser.set_defaults(max_num_seqs=1)

    args: dict = vars(parser.parse_args())
    
    predict_on_data(args, sample_size=10)
