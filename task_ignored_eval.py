    
import os
from pathlib import Path
from combine_metrics import generate_table
from evaluate_metrics import TableMetricsEvaluator
from table_selection import PUBTABNET_JSONL

max_threads = int(os.getenv("MAX_THREADS", "4"))
results_path = Path("/data/brussel/vo/000/bvo00018/vsc11306/cross-modal_experiments/PubTabNet/results")
models = [name for name in os.listdir(results_path) if os.path.isdir(os.path.join(results_path, name))]
task = "transpose"

evaluator = TableMetricsEvaluator(
        structure_only=bool(int(os.getenv("STRUCTURE_ONLY", "0"))),
        ignored_nodes=[],
    )
for model in models:
    model_dir = Path(results_path) / model
    output_dir_track4 = Path(model_dir) / task / "visual_processed_samples.jsonl"
    print(f"Evaluating results from {output_dir_track4} against {PUBTABNET_JSONL}")
    metrics_output = model_dir / task / "ignored_metrics.jsonl"
    evaluation = evaluator.evaluate_jsonl_files(
        pred_jsonl=str(output_dir_track4),
        gt_jsonl=PUBTABNET_JSONL,
        output_path=str(metrics_output),
        max_workers=max_threads,
        matching_by_imgid=bool(int(os.getenv("MATCHING_BY_IMGID", "1"))),
    )
    print(f"Evaluation metrics saved to: {metrics_output}")
    print(f"Matched entries: {evaluation['metrics']['matched']}")

combined_output = Path(__file__).with_name("tex_tables") / f"{task}_ignored.tex"
metrics_filename = "ignored_metrics.jsonl"


table = generate_table(
    results_dir=results_path,
    task=task,
    metrics_filename=metrics_filename
)
combined_output.parent.mkdir(parents=True, exist_ok=True)
combined_output.write_text(table + "\n", encoding="utf-8")
print(f"Combined LaTeX table saved to: {combined_output}")