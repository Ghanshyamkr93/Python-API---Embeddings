import openai
import os
from langchain import OpenAI
from gpt_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from flask import Flask
from flask_restful import Api
import re


app = Flask(__name__)
api = Api(app)

#path = "https://api.openai.com/v1/completions"
  

#def env(apikey):
os.environ["OPENAI_API_KEY"] = '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'

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


@app.route('/testapi2/<input>')
def query(input):
       #akey = env(apikey)
       req = chatbot(input)
       if re.search("is not related to the context information provided.", req):
            response = openai.Completion.create(
                  model="text-davinci-003",
                  prompt=input,
                  temperature=0.7,
                  max_tokens=256,
                  top_p=1,
                  frequency_penalty=0,
                  presence_penalty=0
                  )
            return response.choices[0].text
       elif re.search("not mentioned in the context information provided", req):
             response = openai.Completion.create(
                   model="text-davinci-003",
                   prompt=input,
                   temperature=0.7,
                   max_tokens=256,
                   top_p=1,
                   frequency_penalty=0,
                   presence_penalty=0
                   )
             return response.choices[0].text
       else:
             return req
      

if __name__ == "__main__":
    app.run()