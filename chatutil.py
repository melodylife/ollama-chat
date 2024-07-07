# ChatTTS-01.py
 
import ChatTTS
import torch
import torchaudio
import soundfile
import time
import torch
import random

from modelscope import snapshot_download

module_name = "chatutil"

# do use full path to set the model path. for example "/Users/helloman/assets" . 
MODELPATH = "~/.cache/huggingface/hub/models--2Noise--ChatTTS/snapshots/cc14302f34d7855eb3420d1fd48345012ff1460d/asset"
 
class ChatTTSUtil:
    def __init__(self ,
                 modelPath = MODELPATH,
                 saveFilePath = "output/" ,
                 fixSpkStyle = True):
        self.modelPath = modelPath
        self.wavfilePath = saveFilePath
        self.fixSpkStyle = fixSpkStyle
        self.chat = ChatTTS.Chat()
        self.chat.load_models(local_path = modelPath)
        self.params_refine_text = {"prompt": "[oral_0][laugh_0][break_0]"}
        # Config the speech style with random generation
        std , mean = torch.load(f"{MODELPATH}/spk_stat.pt").chunk(2)
        rand_spk = torch.randn(768) * std + mean
        self.params_infer_code = {
            "spk_emb": rand_spk,
            "temperature": .3,
            "top_P": 0.7,
            "top_K": 20,
            "prompt": "[speed_5]"
        }

    def setRefineTextConf(self , oralConf = "[oral_0]" , laughConf = "[laugh_0]" , breakConf = "[break_0]"):
        self.params_refine_text = {"prompt": f"{oralConf}{laughConf}{breakConf}"}

    def setInferCode(self , temperature = 0.3 , top_P = 0.7 , top_K = 20 , speed = "[speed_5]"):
        self.params_infer_code["temperature"] = temperature
        self.params_infer_code["top_P"] = top_P
        self.params_infer_code["top_K"] = top_K
        self.params_infer_code["prompt"] = speed

    def generateSound(self , texts , savePath = "output/" , filePrefix = "output"):
        wavs = self.chat.infer(texts , use_decoder = True , params_refine_text = self.params_refine_text , params_infer_code = self.params_infer_code)
        wavFilePath = []
        for (index, wave) in enumerate(wavs):
            soundfile.write(f"{savePath}{filePrefix}{index}.wav" , wave[0] , 24000)
            wavFilePath.append(f"{savePath}{filePrefix}{index}.wav")
        return wavFilePath

if __name__ == "__main__":
    chUtil = ChatTTSUtil()
    texts = [
        "大家好，我是Chat T T S，欢迎来到畅的科技工坊。",
        "太棒了，我竟然是第一位嘉宾。",
        "我是Chat T T S， 是专门为对话场景设计的文本转语音模型，例如大语言助手对话任务。我支持英文和中文两种语言。最大的模型使用了10万小时以上的中英文数据进行训练。目前在huggingface中的开源版本为4万小时训练且未S F T 的版本。",
    "耶，我们开始吧"
    ]
    chUtil.setInferCode(0.8 , 0.7 , 20 , speed = "[speed_3]")
    chUtil.generateSound(texts)
