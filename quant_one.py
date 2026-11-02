"""Shrinks one ONNX model to 8-bit weights, in its own process so it has the machine's memory to itself.
usage: python quant_one.py SRC DST OP1,OP2,..."""
import os, shutil, sys, tempfile
import onnx
import onnxruntime.quantization.quant_utils as qu
import onnxruntime.quantization.onnx_quantizer as oq
from onnxruntime.quantization import quantize_dynamic, QuantType

def keep_one_copy(model):
    # The quantizer would re-save and reload the whole model here (fails over 2 GB, needs extra copies).
    qu.add_infer_metadata(model)
    return model
qu.save_and_reload_model_with_shape_infer = keep_one_copy
oq.save_and_reload_model_with_shape_infer = keep_one_copy

src, dst, ops = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
work = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(src)))
big = os.path.join(work, "model.onnx")
quantize_dynamic(src, big, weight_type=QuantType.QUInt8, per_channel=False, op_types_to_quantize=ops,
                 use_external_data_format=True, extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
m = onnx.load(big, load_external_data=True)
# What got shrunk (for the log).
from collections import Counter
ops_count = Counter(n.op_type for n in m.graph.node)
by_type = Counter()
for t in m.graph.initializer:
    by_type[onnx.TensorProto.DataType.Name(t.data_type)] += len(t.raw_data) if t.raw_data else 0
print("ops:", {k: v for k, v in ops_count.items() if k in ("Conv", "ConvInteger", "MatMul", "MatMulInteger", "Gemm", "DynamicQuantizeLinear")}, flush=True)
print("weights MB by type:", {k: v // 1_000_000 for k, v in by_type.items()}, flush=True)
for f in (dst, dst + ".data"):
    if os.path.exists(f): os.remove(f)
if sum(by_type.values()) < 1_900_000_000:
    onnx.save(m, dst, save_as_external_data=False)
else:
    # Over protobuf's 2 GB limit: weights go in one side file next to the model (ONNX Runtime finds it).
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(dst) + ".data", size_threshold=1024)
shutil.rmtree(work, ignore_errors=True)
print("quantized", dst, os.path.getsize(dst) // 1_000_000, "MB", flush=True)
