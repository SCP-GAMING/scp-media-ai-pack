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
onnx.save(m, dst, save_as_external_data=False)  # under 2 GB once shrunk
shutil.rmtree(work, ignore_errors=True)
print("quantized", dst, os.path.getsize(dst) // 1_000_000, "MB", flush=True)
