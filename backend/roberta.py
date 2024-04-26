from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from fastapi import FastAPI

app = FastAPI()
model_name = "deepset/roberta-base-squad2"

question = 'Why is model conversion important?'
context = 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'

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