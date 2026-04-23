import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.autodl_env import apply_autodl_data_disk_env
from shared.csn_paths import default_csn_java_dir, default_eval_models_parent

apply_autodl_data_disk_env()

from scripts.csn_data import load_csn_dataset
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
import torch

_java = default_csn_java_dir()
_test_path = _java / "test.jsonl"
if not _test_path.is_file():
    raise FileNotFoundError(
        f"Not found: {_test_path}; run download_codesearchnet.py (or set CSN_OUTPUT_DIR) first."
    )

test = load_csn_dataset(_test_path, 1000)

_cache_dir = default_eval_models_parent(None) / "1"
_npzs = sorted(_cache_dir.glob("csn_retriever_emb_*.npz"), key=lambda p: p.stat().st_mtime)
if not _npzs:
    raise FileNotFoundError(
        f"No csn_retriever_emb_*.npz under {_cache_dir}; run scripts/evaluate_code_search.py to build index cache first."
    )
data = np.load(_npzs[-1])
embs = data["embeddings"]

tok = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
model = RobertaModel.from_pretrained("microsoft/unixcoder-base")


def encode(text):
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length")
    with torch.no_grad():
        out = model(**inputs)
        vec = out.last_hidden_state[:, 0, :]
        vec = torch.nn.functional.normalize(vec, p=2, dim=1)
        return vec.numpy().reshape(-1)


q = encode(test[0]["nl_query"]).astype(np.float32)
q = q / (np.linalg.norm(q) + 1e-8)

sims = embs @ q
idx = np.argsort(-sims)[:5]

print("Query:", test[0]["nl_query"])
print("Target URL:", test[0]["url"])
print("Top 5 URLs:", [test[i]["url"] for i in idx])
print("Top 5 Sims:", sims[idx])

target_idx = 0
print("Target Sim:", sims[target_idx])
print("Target Rank:", np.where(np.argsort(-sims) == target_idx)[0][0])
