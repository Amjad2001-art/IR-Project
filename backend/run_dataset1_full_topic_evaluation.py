import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from services.evaluation_service import evaluate_all_methods_with_topic_detection
from services.model_loader_service import ModelLoaderService


OUT_PATH = ROOT.parent / "report_assets" / "topic_evaluation_results_dataset1_full.json"

METHODS = [
    "tfidf",
    "word2vec",
    "bm25",
    "inverted_index",
    "serial_hybrid",
    "parallel_hybrid",
]


def main():
    OUT_PATH.parent.mkdir(exist_ok=True)
    save_dir = str(ROOT / "saved_files")
    loader = ModelLoaderService(save_dir=save_dir)
    loaded_data = loader.load_all()

    results = {
        "dataset1": evaluate_all_methods_with_topic_detection(
            dataset="dataset1",
            methods=METHODS,
            loaded_data=loaded_data,
            top_k=10,
            max_queries=None,
            k1=1.5,
            b=0.75,
            alpha=0.6,
            save_dir=save_dir,
        )
    }

    with open(OUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(OUT_PATH)


if __name__ == "__main__":
    main()
