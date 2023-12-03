from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import login

# Here, put your own Hugging Face token copied from https://huggingface.co/settings/tokens
my_token = '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'
login(token=my_token)

model = "meta-llama/Llama-2-70b-chat-hf" # meta-llama/Llama-2-7b-hf

tokenizer = AutoTokenizer.from_pretrained(model, token=my_token)

from transformers import pipeline

llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)


def get_llama_response(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    print("Chatbot:", sequences[0]['generated_text'])

prompt = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
get_llama_response(prompt)

prompt = """I'm a programmer and Python is my favorite language because of it's simple syntax and variety of applications I can build with it.\
Based on that, what language should I learn next?\
Give me 5 recommendations"""
get_llama_response(prompt)

prompt = 'How to learn fast?\n'
get_llama_response(prompt)

prompt = 'I love basketball. Do you have any recommendations of team sports I might like?\n'
get_llama_response(prompt)

prompt = 'How to get rich?\n'
get_llama_response(prompt)

