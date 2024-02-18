from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

llm = OpenAI(
    server_url = "http://localhost:8080"
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "How can i create a simple test case in Golang"

res = llm_chain.run(question)

print(res)