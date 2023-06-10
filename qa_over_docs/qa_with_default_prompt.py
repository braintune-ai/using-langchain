import os
from uuid import uuid4

import openai
import requests
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.callbacks import get_openai_callback, OpenAICallbackHandler
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import DeepLake

load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
SESSION_ID = str(uuid4())


openai.api_key = os.getenv("OPENAI_API_KEY")
active_loop_org = os.getenv("ACTIVELOOP_ORG")


def register_prompt_to_braintune(prompt: PromptTemplate):
    url = f"{BASE_URL}/v0/prompts/langchain_template"
    headers = {"api_key": os.getenv("BRAINTUNE_API_KEY")}
    response = requests.post(url, headers=headers, json=prompt.dict())
    response.raise_for_status()
    registered_prompt = response.json()
    return registered_prompt


def send_event(registered_prompt: dict, cb: OpenAICallbackHandler, llm: OpenAI):
    llm_parameters = llm.dict(exlude={"request_timeout", "logit_bias", "_type"})
    payload = {
        "prompt_id": registered_prompt["id"],
        "prompt_version": registered_prompt["active_version"],
        "model_name": llm.model_name,
        "parameters": llm_parameters,
        "completion_tokens": cb.completion_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "session_id": SESSION_ID,
    }
    url = f"{BASE_URL}/v0/events/openai/simple"
    headers = {"api_key": os.getenv("BRAINTUNE_API_KEY"), "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    if response.status_code == 202:
        return response.json()
    else:
        raise Exception(f"Error sending event: {response.text}")


dataset_path = f"hub://{active_loop_org}/data"
embeddings = OpenAIEmbeddings()

db = DeepLake(dataset_path=dataset_path, read_only=True, embedding_function=embeddings)

retriever = db.as_retriever()
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["k"] = 4

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=False)
qa.verbose = True


prompt_template = qa.combine_documents_chain.llm_chain.prompt

registered_prompt = register_prompt_to_braintune(prompt_template)
print(f"Prompt `{registered_prompt['name']}` has been registered")
print(f"You can access it at https://braintune.ai/prompts/{registered_prompt['id']}")

while True:
    query = input("Enter query:")
    with get_openai_callback() as openai_callback:
        ans = qa({"query": query})
        send_event(registered_prompt, openai_callback, qa.combine_documents_chain.llm_chain.llm)

    print(ans["result"])
