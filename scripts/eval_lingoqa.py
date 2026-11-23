"""LingoQA (paper Table 4): the model answers the 500 validation questions itself, zero-shot.

Alpamayo 1.5's native VQA interface (`helper.create_vqa_message`) is used; LingoQA gives 5 front-camera
frames per segment, the last 4 are passed as camera index 1 ("Front camera"). Greedy decoding, at most
128 new tokens, stop at <|answer_end|>. Writes <out>/rows.json and <out>/predictions.csv (the columns
LingoQA's evaluate.py reads); score with scripts/score_lingoqa.py.

Data: the LingoQA validation set (https://github.com/wayveai/LingoQA, Wayve non-commercial licence;
not redistributed here) with `val.parquet` and `images/val/<segment_id>/{0..4}.jpg` under $LINGOQA_DATA.

  LINGOQA_DATA=/path/to/lingoqa/val python scripts/eval_lingoqa.py --model runs/moop_u40 --out runs/lingoqa/moop_u40
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from moop import MODEL_REV
from moop.model_io import load_base_model, set_attn_impl

FRONT_CAMERA, N_FRAMES = 1, 4


def lingo_root():
    return Path(os.environ["LINGOQA_DATA"])


def load_questions():
    """500 unique (question_id, segment_id) with their 2 human references, sorted by segment."""
    df = pd.read_parquet(lingo_root() / "val.parquet")
    g = df.groupby(["question_id", "segment_id", "question"])["answer"].apply(list).reset_index()
    q = [{"question_id": r.question_id, "segment_id": r.segment_id, "question": r.question,
          "references": list(r.answer)} for r in g.itertuples()]
    return sorted(q, key=lambda x: (x["segment_id"], x["question_id"]))


def load_frames(segment_id):
    d = lingo_root() / "images" / "val" / segment_id
    frames = [np.transpose(np.asarray(Image.open(d / f"{i}.jpg").convert("RGB")), (2, 0, 1))
              for i in range(5 - N_FRAMES, 5)]  # last 4 of 5, (3, H, W) each
    return torch.from_numpy(np.stack(frames))[None]  # (1, 4, 3, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="'dense' or a directory holding slim_meta.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gen", type=int, default=128)
    args = ap.parse_args()

    from alpamayo1_5 import helper
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.model == "dense":
        model, processor = load_base_model("cuda")
    else:
        from moop.slim import load_slim
        model = load_slim(args.model, device="cuda")
        processor = helper.get_processor(model.tokenizer)
    set_attn_impl(model, "sdpa")
    tok = model.tokenizer
    gc = model.vlm.generation_config
    gc.do_sample, gc.temperature, gc.top_p, gc.top_k = False, None, None, None
    gc.num_return_sequences, gc.max_new_tokens = 1, args.max_gen
    gc.output_logits, gc.return_dict_in_generate = False, True
    gc.pad_token_id, gc.eos_token_id = tok.pad_token_id, tok.convert_tokens_to_ids("<|answer_end|>")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        "model": args.model, "model_revision": MODEL_REV, "decoding": "greedy", "max_gen": args.max_gen,
        "frames": N_FRAMES, "camera_index": FRONT_CAMERA, "gpu": torch.cuda.get_device_name(0)}, indent=2))
    rows_path = out / "rows.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    done = {r["question_id"] + "|" + r["segment_id"] for r in rows}
    questions = load_questions()
    seg, frames = None, None
    for i, q in enumerate(questions):
        if q["question_id"] + "|" + q["segment_id"] in done:
            continue
        if q["segment_id"] != seg:
            seg, frames = q["segment_id"], load_frames(q["segment_id"]).to("cuda")
        msgs = helper.create_vqa_message(frames=frames.flatten(0, 1), question=q["question"],
                                         camera_indices=torch.tensor([FRONT_CAMERA], device="cuda"))
        enc = processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                                            continue_final_message=True, return_dict=True, return_tensors="pt")
        enc = helper.to_device(dict(enc), "cuda")
        ids = enc.pop("input_ids")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.vlm.generate(input_ids=ids, generation_config=gc, **enc).sequences[0, ids.shape[1]:]
        answer = tok.decode(gen, skip_special_tokens=True).strip()
        rows.append({"question_id": q["question_id"], "segment_id": q["segment_id"], "question": q["question"],
                     "answer": answer, "references": q["references"], "gen_len": int(gen.shape[0]),
                     "truncated": int(gen.shape[0]) >= args.max_gen})
        if len(rows) % 25 == 0:
            rows_path.write_text(json.dumps(rows, indent=2))
        print(f"[{i + 1}/{len(questions)}] {answer[:80]!r}", flush=True)
    rows_path.write_text(json.dumps(rows, indent=2))
    pd.DataFrame([{k: r[k] for k in ("question_id", "segment_id", "answer")} for r in rows]).to_csv(
        out / "predictions.csv", index=False)


if __name__ == "__main__":
    main()
