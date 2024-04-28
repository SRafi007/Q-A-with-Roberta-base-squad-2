from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi import FastAPI

app = FastAPI()
model_name = "deepset/roberta-base-squad2"

question = 'Why is bangladesh  important?'
context = 'Bangladesh officially the People's Republic of Bangladesh,[b] is a country in South Asia. It is the eighth-most populous country in the world and is among the most densely populated countries with a population of nearly 170 million in an area of 148,460 square kilometres (57,320 sq mi). Bangladesh shares land borders with India to the north, west, and east, and Myanmar to the southeast. To the south, it has a coastline along the Bay of Bengal. It is narrowly separated from Bhutan and Nepal by the Siliguri Corridor, and from China by the mountainous Indian state of Sikkim in the north. Dhaka, the capital and largest city, is the nation's political, financial, and cultural centre. Chittagong, the second-largest city, is the busiest port on the Bay of Bengal. The official language of Bangladesh is Bengali.'

# Load model & tokenizer
#model = AutoModelForQuestionAnswering.from_pretrained(model_name)
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

def que_ans(question, context):
    res = nlp(question=question, context=context)
    return res


@app.get("/") 
async def get_answer(question, context):
    return que_ans(question, context)
