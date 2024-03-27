from fastapi import FastAPI, HTTPException
import uvicorn
import gunicorn
from fastapi.middleware.cors import CORSMiddleware
from roberta import que_ans

app =FastAPI()

origins=['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']

)

@app.get("/answer/{contex}/{question}")
def answers(question, context):
    result=que_ans(question, context)
    if not result:
        raise HTTPException(status_code=400)
    return result