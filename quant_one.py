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
    n = len(t.raw_data) + 4 * (len(t.float_data) + len(t.int32_data)) + 8 * len(t.int64_data)
    by_type[onnx.TensorProto.DataType.Name(t.data_type)] += n
const_bytes = sum(len(a.t.raw_data) + 4 * len(a.t.float_data) for nd in m.graph.node if nd.op_type == "Constant" for a in nd.attribute if a.type == onnx.AttributeProto.TENSOR)
used = {i for nd in m.graph.node for i in nd.input}
unused = [t for t in m.graph.initializer if t.name not in used]
print("ops:", {k: v for k, v in ops_count.items() if k in ("Conv", "ConvInteger", "MatMul", "MatMulInteger", "Gemm", "DynamicQuantizeLinear")}, flush=True)
print("weights MB by type:", {k: v // 1_000_000 for k, v in by_type.items()}, "constants MB:", const_bytes // 1_000_000, "unused initializers:", len(unused), flush=True)
for t in unused:
    m.graph.initializer.remove(t)
for f in (dst, dst + ".data"):
    if os.path.exists(f): os.remove(f)
try:
    onnx.save(m, dst, save_as_external_data=False)
except Exception as e:
    print("one file too big, using a side file:", type(e).__name__, flush=True)
    if os.path.exists(dst): os.remove(dst)
    # Weights go in one side file next to the model (ONNX Runtime finds it by name).
    onnx.save(m, dst, save_as_external_data=True, all_tensors_to_one_file=True,
              location=os.path.basename(dst) + ".data", size_threshold=1024, convert_attribute=True)
shutil.rmtree(work, ignore_errors=True)
print("quantized", dst, (os.path.getsize(dst) + (os.path.getsize(dst + ".data") if os.path.exists(dst + ".data") else 0)) // 1_000_000, "MB", flush=True)
