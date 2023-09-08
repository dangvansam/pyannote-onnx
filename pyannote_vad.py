import os
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
            vad_model_url = "https://sync.admicro.vn/f/a8903c52b07b4a7fa8fc/?dl=1"
            save_model_path = "/tmp/pyannot_vad_model.bin"
            if not os.path.exists(save_model_path):
                os.system(f"wget -q --show-progress --progress=bar:force {vad_model_url} -O {save_model_path}")
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

    def split(self, filepath: str):
        audio, sr = soundfile.read(filepath)

        if len(audio.shape) > 1:    # if audio have multi channel
            audio = audio[0]
        
        output = self.vad_pipeline(filepath)
        # del vad_pipeline
        
        speech_timestamps = []
        for speech in output.get_timeline().support():
            speech_timestamps.append({
                "start": int(float(speech.__dict__["start"]) * sr),
                "end": int(float(speech.__dict__["end"]) * sr)
            })
        return audio, speech_timestamps
    
if __name__=='__main__':
    from time import perf_counter
    vad_model = PyannoteVAD(model="onnx_model/pyannot_vad_model.onnx")
    # vad_model = PyannoteVAD()
    total_cost = 0
    for _ in range(10):
        st = perf_counter()
        _, rs = vad_model.split("D:/quay-vc-short.wav")
        print(len(rs))
        cost = perf_counter() - st
        total_cost += cost
        print(f"time cost = {cost}")
    logger.info(f"mean time cost = {total_cost/10}")