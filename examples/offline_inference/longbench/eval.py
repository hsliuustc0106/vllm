import os, json, argparse
import numpy as np
from task_templates import TASK_TEMPLATES

from metrics import (
    qa_f1_score,
    rouge_score,
    classification_score,
    retrieval_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def evaluate_on_dataset(args):
    score_path = os.path.join(f"scores_{args.method_name}" if args.method_name != "" else "scores")
    os.makedirs(score_path, exist_ok=True)
    pred_file = os.path.join(f"{args.pred_file_name}.jsonl")
    score_file = os.path.join(score_path, f"{os.path.basename(pred_file).split('.jsonl')[0]}_score.jsonl")

    with open(pred_file, "r", encoding="utf-8") as f:
        predictions, answers = [], []
        comments = []
        for line in f:
            data = json.loads(line)
            predictions.append(data["pred"])
            answers.append(data["answers"])
            if "all_classes" in data:
                all_classes = data["all_classes"]
            else:
                all_classes = None
            if "comments" in data:
                comments.append(data["comments"])
            else:
                comments.append(None)
    print(f"Loaded {len(predictions)} predictions from {pred_file}")

    print(f"Scoring {args.dataset} predictions from {args.model_id}...")
    total_score = 0.
    for (idx, prediction, ground_truths, comment) in zip(range(len(predictions)), predictions, answers, comments):
        score = 0.
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[args.dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score

        print(f"Sample {idx} - Score: {round(100 * score, 2)}")
        json_data = {
            "score": round(100 * score, 2),
            **({"comments": comment} if comment else {}),
            "pred": prediction,
            "answers": ground_truths,
        }
        with open(score_file, "a", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False)
            f.write('\n')

    avg_score = round(100 * total_score / len(predictions), 2)
    os.rename(score_file, f"{score_file.split('.jsonl')[0]}_{avg_score}.jsonl")

    print(f"Average score of {len(predictions)} samples for {args.dataset} in prediction file {pred_file} on {args.model_id}: {avg_score}")
    return avg_score, score_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model ID to evaluate")
    parser.add_argument("--dataset", type=str, default="qasper", help="Dataset to evaluate on")
    parser.add_argument("--datasets", type=str, nargs="+", default=None, help="List of datasets to evaluate on, if None, evaluate on the specified dataset")
    parser.add_argument("--method-name", type=str, default="", help="Name of the method to evaluate")
    parser.add_argument("--pred-file-name", type=str, default="", help="Prediction file name to evaluate")
    args = parser.parse_args()

    if args.datasets is not None:
        all_tasks = False
        if "all" in args.datasets:
            args.datasets = list(TASK_TEMPLATES.keys())
            all_tasks = True
        avg_scores = []
        for dataset in args.datasets:
            args.dataset = dataset
            avg_score, score_path = evaluate_on_dataset(args)
            avg_scores.append(avg_score)
        tasks_avg_score = np.mean(avg_scores)
        print(f"Average score for {args.datasets} on {args.model_id} "
              f"{'with method ' + args.method_name if args.method_name else ''}: {tasks_avg_score:.2f}")
        if all_tasks:
            new_score_path = f"{score_path}_{tasks_avg_score:.2f}"
            if not os.path.exists(new_score_path):
                os.rename(score_path, new_score_path)
    else:
        evaluate_on_dataset(args)