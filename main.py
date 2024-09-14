from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import wordnet
from transformers import pipeline
from Model.inference_pos import get_type_word
from Model.inference_antonym import get_word_antonym

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/arabic_translate')
async def text_translation(text: str):
    source_language = "eng_Latn"
    target_language = "arz_Arab"
    pipe = pipeline("translation", model="NLLB_checkpoint")
    result = pipe(text, src_lang=source_language, tgt_lang=target_language)
    if result and isinstance(result, list) and 'translation_text' in result[0]:
        return result[0]['translation_text']
    else:
        raise ValueError("Translation failed or returned an unexpected result.")

@app.get('/get_Word_type')
async def get_word_type(word: str):
    return get_type_word(word)

@app.get('/get_Word_antonym')
async def get_antonym(word: str):
    return get_word_antonym(word)


def get_word_details(word: str):
    syn = wordnet.synsets(word)
    word_name = word
    pos = get_type_word(word) 
    definition = syn[0].definition()
    example = syn[0].examples()
    antonyms = get_word_antonym(word)  
    translated2arabic = text_translation(word) 
    word_description = [word_name, definition, antonyms, pos, translated2arabic, example]
    return word_description

@app.post("/get_word_details", response_class=HTMLResponse)
async def define_word(request: Request):
    form_data = await request.form()
    word = form_data.get("word", "").lower()
    word_details = await get_word_details(word) 
    word_name = word_details[0]
    definition = word_details[1]
    antonyms = word_details[2]
    word_type = word_details[3]
    arabic_translation = word_details[4]
    example = word_details[5]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "word": word_name,
        "definition": definition,
        "antonyms": antonyms,
        "word_type": word_type,
        "arabic_translation": arabic_translation,
        "example": example
    })
