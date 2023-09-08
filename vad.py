import os
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
from loguru import logger
import soundfile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio import Model
import GPUtil

def auto_select_device()->torch.device:
    gpus = GPUtil.getGPUs()
    if gpus:
        gpus.sort(key=lambda gpu: gpu.memoryUsed, reverse=False)
        logger.info(f"Name: {gpus[0].name}, ID: {gpus[0].id}")
        logger.info(f"Memory Used: {gpus[0].memoryUsed} MB")
        device = torch.device(gpus[0].id)
    else:
        device = torch.device('cpu')
        logger.info(f"Name: CPU")
    return device

class PyannoteVAD:
    def __init__(self, model: str = None, batch_size: int = 32):
        if model is None:
            # vad_model_url = "https://huggingface.co/pyannote/segmentation/blob/main/pytorch_model.bin"
            save_model_path = "pytorch_model/vad_model.bin"
            if not os.path.exists(save_model_path):
                logger.error("model not found, download model from https://huggingface.co/pyannote/segmentation/blob/main/pytorch_model.bin")
                # os.system(f"wget -q --show-progress --progress=bar:force {vad_model_url} -O {save_model_path}")
                exit()
            model = Model.from_pretrained(checkpoint=save_model_path)

        vad_pipeline = VoiceActivityDetection(segmentation=model, batch_size=batch_size)
        device = auto_select_device()
        vad_pipeline = vad_pipeline.to(device)

        HYPER_PARAMETERS_2 = {
            "onset": 0.8,
            "offset": 0.5,
            "min_duration_on": 0.5,
            "min_duration_off": 0.1
        }
        vad_pipeline.instantiate(HYPER_PARAMETERS_2)
        self.vad_pipeline = vad_pipeline

    def get_vad_model(self) -> VoiceActivityDetection:
        return self.vad_pipeline

    def split(
        self,
        filepath: str,
        output_type: str = "sample"
    ) -> Tuple[np.ndarray, List[Dict[str, Union[float, int]]]]:
        """perform voice activity detection on audio file

        Args:
            filepath (str): audio input file with wav format
            output_type (str, optional): output type ["sample", "second"]

        Returns:
            audio (np.ndarray): audio samples
            speech_timestamps (list): detected speech segments
        """
        audio, sr = soundfile.read(filepath)
       
        if len(audio.shape) > 1:
            audio = audio[0]
        
        output = self.vad_pipeline(filepath)
        
        speech_timestamps = []
        for speech in output.get_timeline().support():
            if output_type == "sample":
                speech_timestamps.append({
                    "start": int(float(speech.__dict__["start"]) * sr),
                    "end": int(float(speech.__dict__["end"]) * sr)
                })
            else:
                speech_timestamps.append({
                    "start": float(speech.__dict__["start"]),
                    "end": float(speech.__dict__["end"])
                })
        return audio, speech_timestamps
    
if __name__=='__main__':
    import argparse
    from time import perf_counter
    
    parser = argparse.ArgumentParser(description="export onnx model")
    parser.add_argument("-m", "--model", required=True, help="vad model path")
    parser.add_argument("-i", "--wav", required=True, help="input wav path")
    args = parser.parse_args()
    
    assert os.path.exists(args.model), f"{args.model} not found"
    assert os.path.exists(args.wav), f"{args.wav} not found"
    
    vad_model = PyannoteVAD(model=args.model)

    n_tests = 10
    total_cost = 0
    for _ in range(n_tests):
        st = perf_counter()
        _, rs = vad_model.split(args.wav)
        logger.info(f"num segments: {len(rs)}")
        cost = perf_counter() - st
        total_cost += cost
        logger.info(f"time cost = {cost}")
    
    print(f"vad result: {rs}")
    
    logger.info(f"mean time cost = {total_cost/n_tests}")