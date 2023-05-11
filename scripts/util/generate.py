"""
Module that uses a Large Language Model as a library or service 
to generate text given a prompt. Currently, "generate" functions
relies on publicly available pre-trained models, and so using it
requires a machine with GPUs. If you want to use a service instead,
you need to implement the "generate" functions using API calls.
"""
import time
import logging
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


MODEL_NAME = "allenai/tk-instruct-3b-def"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.info("Loading model: %s", MODEL_NAME)
start_time = time.time()
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)
logging.info("Took: %f seconds.", time.time() - start_time)


def generate_answer_regular(
    prompt, do_sample=True, temparature=0.01, top_p=0.4, top_k=50, max_new_tokens=3
):
    """
    Takes in a "prompt" (text) and uses an LLM to generate text using the specified
    parameters. Only the generated text is returned in the output, in lowercase.
    Tested on "EleutherAI/gpt-neo-2.7B"
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    gen_tokens = model.generate(
        input_ids,
        do_sample=do_sample,
        temperature=temparature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    return gen_text[gen_text.find(prompt) + len(prompt) :].strip().lower()


def generate_answer_instruct(prompt):
    """
    Takes in a "prompt" (text) and uses an instruction-tuned LLM to generate an
    answer to an instruction prompt.
    Tested on "allenai/tk-instruct-3b-def"
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(input_ids, max_length=10)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output.strip().lower()
