#!/usr/bin/env python3
"""ONNX fp32 -> fp16 conversion utility (standard onnxconverter_common based).

Used both offline (model preparation) and at runtime by the ADI CLI when the
host does not have enough memory for the fp32 graph: fp32 stays the single
authoritative artifact; fp16 is a derived, cacheable conversion.

Design notes:
- keep_io_types=True: graph inputs/outputs remain fp32 (Cast nodes inserted
  internally), so the C++ side keeps feeding float tensors regardless of the
  weight dtype. This is what makes runtime precision switching transparent.
- Large models are saved with external data (single .onnx_data file), matching
  the layout produced by the optimum export chain.
- Every converted model is verified by actually loading it with ONNX Runtime;
  a conversion that cannot be loaded is reported as failure (exit 1).

Usage:
  python3 onnx_fp16_convert.py <src_model_dir_or_onnx> <dst_dir> [--no-verify]
"""
import os
import shutil
import sys

import onnx


def convert_one(src_onnx: str, dst_dir: str, verify: bool = True) -> bool:
    name = os.path.basename(os.path.dirname(src_onnx)) or os.path.basename(src_onnx)
    os.makedirs(dst_dir, exist_ok=True)
    dst_onnx = os.path.join(dst_dir, "model.onnx")

    print(f"[convert] {src_onnx} -> {dst_onnx}", flush=True)

    # Path-based shape inference first (sidecar next to the source so external
    # data refs resolve): multi-GB exports blow past the 2GB protobuf cap and
    # in-memory infer_shapes silently yields an EMPTY model on that path.
    # Without value_info, dtype propagation is incomplete and MatMul ends up
    # with mixed float/float16 inputs that ORT rejects.
    src_dir = os.path.dirname(src_onnx) or "."
    tmp_inferred = os.path.join(src_dir, ".model_inferred_tmp.onnx")
    onnx.shape_inference.infer_shapes_path(src_onnx, tmp_inferred)
    model = onnx.load(tmp_inferred)
    os.remove(tmp_inferred)

    # Use ONNX Runtime's own transformer-oriented converter (NOT the bare
    # onnxconverter_common one): it correctly re-routes casts around blocked
    # ops. optimum exports carry fp32 constants inside LayerNorm inputs, and a
    # naive conversion leaves mixed float/float16 signatures that ORT rejects.
    # LayerNorm stays fp32 — standard diffusion practice (numerically
    # sensitive, cheap relative to matmuls).
    from onnxruntime.transformers import float16 as ort_float16
    model_fp16 = ort_float16.convert_float_to_float16(
        model,
        keep_io_types=True,        # C++ feeds float32 tensors; casts inserted inside
        op_block_list=["LayerNormalization"],
        disable_shape_infer=True,  # already inferred above; also avoids the 2GB cap
    )
    onnx.save_model(
        model_fp16,
        dst_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=1024,
        convert_attribute=False,
    )
    del model, model_fp16

    if verify:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            sess = ort.InferenceSession(dst_onnx, sess_options=so,
                                        providers=["CPUExecutionProvider"])
            print(f"[verify] OK ({len(sess.get_inputs())} inputs)", flush=True)
            del sess
        except Exception as e:  # noqa: BLE001
            print(f"[verify] FAIL: {str(e)[:300]}", flush=True)
            return False
    return True


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src = sys.argv[1]
    dst = sys.argv[2]
    verify = "--no-verify" not in sys.argv

    src_onnx = src if src.endswith(".onnx") else os.path.join(src, "model.onnx")
    if not os.path.exists(src_onnx):
        print(f"[error] source not found: {src_onnx}")
        return 2
    ok = convert_one(src_onnx, dst, verify=verify)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
