import openai
import os
from langchain import OpenAI
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from flask import Flask, jsonify, make_response
from flask_restful import Api
import re
import requests
import json
import jsonpickle
import base64


app = Flask(__name__)
api = Api(app)


endpoint = "https://gpt3prototype.openai.azure.com/openai/deployments/GPT_turbo/completions?api-version=2022-12-01"
headers = {"Content-Type": "application/json", "api-key": "f089541dc0a74f2ca5bccc085ab6bee9"}


#def env(apikey):
os.environ["OPENAI_API_KEY"] = ''

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 20
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
    #llm_predictor = LLMPredictor(llm=openai())
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-embedding-ada-002", max_tokens=num_outputs))
    documents = SimpleDirectoryReader(directory_path).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('index.json')
    return index

def chatbot(input_text):
        index = GPTSimpleVectorIndex.load_from_disk('index.json')
        response = index.query(input_text, response_mode="compact")
        return response.response


index = construct_index("docs")


@app.route('/testapi/<input>')
def query(input):
       #akey = env(apikey)
       req = chatbot(input)
       if re.search("is not related to the context information provided.", req):
            data = {
                   "prompt":"<|im_start|>system\nYou are an AI assistant that helps people find information.\n<|im_end|>\n<|im_start|>user\n"+input+"\n<|im_end|>\n<|im_start|>assistant\n",
                   "max_tokens":800,
                   "temperature":0.7,
                   "frequency_penalty":0,
                   "presence_penalty":0,
                   "top_p":0.95,
                   "stop":["<|im_end|>"]
                   }
            response = requests.post(endpoint, data = data, headers=headers)
            rep = jsonpickle.encode(response, unpicklable=False)
            rep1 = json.dumps(rep)
            rep2 = jsonpickle.decode(rep1)
            rep3 = json.loads(rep2)
            return rep3
       elif re.search("not mentioned in the context information provided", req):
             data = {
                   "prompt":"<|im_start|>system\nYou are an AI assistant that helps people find information.\n<|im_end|>\n<|im_start|>user\n"+input+"\n<|im_end|>\n<|im_start|>assistant\n",
                   "max_tokens":800,
                   "temperature":0.7,
                   "frequency_penalty":0,
                   "presence_penalty":0,
                   "top_p":0.95,
                   "stop":["<|im_end|>"]
             }
             response = requests.post(endpoint, data = data, headers=headers)
             rep = jsonpickle.encode(response, unpicklable=False)
             rep1 = json.dumps(rep)
             rep2 = jsonpickle.decode(rep1)
             rep3 = json.loads(rep2)
             return rep3
       else:
             return req
      

if __name__ == "__main__":
    app.run()
