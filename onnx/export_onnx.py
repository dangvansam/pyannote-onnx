import argparse
import torch
import os
import onnxruntime as ort
from pyannote.audio import Model
from loguru import logger

def main():
    parser = argparse.ArgumentParser(description="export onnx model")
    parser.add_argument("-i", "--pytorch_model", required=True, help="input pytorch model")
    parser.add_argument("-o", "--output_dir", required=True, help="output onnx model dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"export to onnx: {args.pytorch_model} => {args.output_dir}")
    model = Model.from_pretrained(args.pytorch_model)
    print(model)
    print(model.specifications)
    
    model_filename = os.path.basename(args.pytorch_model).replace(".bin", "")
    onnx_filepath = os.path.join(args.output_dir, model_filename + ".onnx")
    model_specifications_filepath = os.path.join(args.output_dir, model_filename + ".pt")
    torch.save(model.specifications, model_specifications_filepath)

    max_length = int(model.specifications.duration * 16000)
    dummy_input = torch.zeros(32, 1, max_length)
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_filepath,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "B", 1: "C", 2: "T"},
        },
    )
    so = ort.SessionOptions()
    so.optimized_model_filepath = onnx_filepath
    # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
    ort.InferenceSession(onnx_filepath, sess_options=so)
    logger.success("export to onnx model success")

if __name__ == "__main__":
    main()
