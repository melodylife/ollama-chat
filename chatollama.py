import wave
import time
from pathlib import Path
from pyaudio import PyAudio, paInt16, paContinue, paComplete
from vosk import KaldiRecognizer, Model as VoskModel, SetLogLevel
from threading import Thread,Lock as ThreadLock
from pynput import keyboard  # pip install pynput
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage


import json
import time

from chatutil import ChatTTSUtil

module_name = "chatollama"

VOSKMODELPATH = "voskmodels/vosk-model-cn-0.22"
OLLAMAMODEL = "gemma2"

class chatOllama:

    # initiate the class with channel, chunk and fps of the audio stream for input.
    def __init__(self , channel = 1 , chunk = 4096 , fps = 44100 , ollamaModel = OLLAMAMODEL , voskModelPath = VOSKMODELPATH , rate = 16000):
        self.pAud = PyAudio()
        self.chunk = chunk
        self.channel = channel
        self.fps = fps
        self.stream = None
        self.flag = False
        self.kill = False
        self.saveFile = "saveAudio/test.mp3"
        self.ollamaModel = ollamaModel
        self.chatLLM = None
        self.rate = rate
        self.voskRec = KaldiRecognizer(VoskModel(voskModelPath) , self.rate)
        #self.pAud = PyAudio()
        self.userMsg = ""
        self.ollamaRes = ""
        self.isEnd = False
        self.lock = ThreadLock()
        self.chatGenerator = ChatTTSUtil()
        self.resFlag = False

    def createStream(self , inputFlag = False , outputFlag = False):
        pAud = PyAudio()
        if inputFlag:
            stream = pAud.open(format = paInt16, channels = self.channel , input = inputFlag, frames_per_buffer = 4096 , rate = self.rate)
        elif outputFlag:
            stream = pAud.open(format = paInt16, channels = self.channel , output = outputFlag , rate = 24000)
        return stream

    def queryOllama(self):
        self.chatLLM = ChatOllama(model=OLLAMAMODEL)
        prompt = ChatPromptTemplate.from_template("Answer the question in simplified Chinese. Don\'t use any emoji in the answer.\n Question: {question}")
        chain = prompt | self.chatLLM  | self.extractResponse
        self.ollamaRes = chain.invoke({"question": self.userMsg})

    def extractResponse(self , ai_message: AIMessage):
        print(ai_message.content)
        return ai_message.content

    def generateAudioResponse(self , texts ,savePath = "output/"):
        return self.chatGenerator.generateSound(texts , savePath = savePath)


    def keyDectect(self):
        with keyboard.Listener(on_press = self.controlRec) as listener:
            listener.join()

    def controlRec(self , key):
        try:
            if key.char == "b":
                print("Start Recording...")
                self.flag = True
                self.execRec()
            elif key.char == "s":
                print("Stop Recording...\n")
                self.flag = False
                self.queryOllama()
                self.resFlag = True
            elif key.char == "e":
                print("Terminate Recording...")
                #self.stream.terminate()
                self.pAud.terminate()
                self.isEnd = True
        except Exception as e:
            print(e)

    def recStream(self):
        #if self.flag:
        self.stream = self.createStream(inputFlag = True)
        self.stream.start_stream()
        print("Please ask you question and press \'s\' to stop recording and send the question to Ollama")
        self.userMsg = ""
        while self.stream.is_active():
            #self.voskRec.SrtResult
            if not self.flag:
                self.stream.stop_stream()
                self.stream.close()
                break
            data: bytes = self.stream.read(4096)
            if self.voskRec.AcceptWaveform(data):
                #self.lock.acquire()
                audioMsg = json.loads(self.voskRec.Result())
                self.userMsg = audioMsg["text"].replace(" " , "")
                #self.lock.release()
            else:
                res = json.loads(self.voskRec.PartialResult())
                if res["partial"]:
                    #print(res["partial"])
                    self.userMsg = self.userMsg + res["partial"]
        voskAudioFinal = json.loads(self.voskRec.FinalResult())
        self.userMsg = voskAudioFinal["text"].replace(" " , "")
        print(f"This is the input audio {self.userMsg}")


    def responseSpeech(self , filePath):
        soundFile = wave.open(filePath , "rb")
        stream = self.createStream(outputFlag = True)
        data = soundFile.readframes(4096)
        while len(data) > 0:
            stream.write(data)
            data = soundFile.readframes(4096)
        print("Reponse Done...")
        stream.stop_stream()
        stream.close()


    def execRec(self):
        thread_recstream = Thread(target = self.recStream)
        thread_recstream.start()

    def run(self):
        thread_keyboard = Thread(target = self.keyDectect , daemon = True)
        thread_keyboard.start()
        while not self.isEnd:
            if self.resFlag == True:
                texts = [self.ollamaRes]
                filePath = self.generateAudioResponse(texts)
                self.responseSpeech(filePath[0])
                self.resFlag = False
            pass

