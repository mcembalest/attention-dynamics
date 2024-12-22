from flask import Flask, Response, request
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread
import json
import torch

RANDOM_SEED = 42 
torch.manual_seed(RANDOM_SEED)

app = Flask(__name__, static_folder='static')

device = "cpu"
max_new_tokens = 350

model_a_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
model_b_id = "HuggingFaceTB/SmolLM2-360M-Instruct"


model_a = None
model_a_tokenizer = None
model_b = None
model_b_tokenizer = None

def init_models():
    global model_a, model_a_tokenizer
    global model_b, model_b_tokenizer
    if model_a is None:
        model_a_tokenizer = AutoTokenizer.from_pretrained(model_a_id)
        model_a = AutoModelForCausalLM.from_pretrained(model_a_id)
        model_a.to(device)
        model_a.eval()
    
    if model_b is None:
        model_b_tokenizer = AutoTokenizer.from_pretrained(model_b_id)
        model_b = AutoModelForCausalLM.from_pretrained(model_b_id)
        model_b.to(device)
        model_b.eval()

def process_attention_scores(attention_scores, prompt_length):
    steps_attention = []
    
    for step_attention in attention_scores:
        # Average across heads for the last layer
        layer_attns = step_attention[-1].squeeze(0)  # Remove batch dimension
        # Average across attention heads
        attns_per_head = layer_attns.mean(dim=0)  # Average across heads
        
        # Get attention to prompt tokens only, excluding the first token (null attention)
        # and the generated token
        prompt_attention = torch.zeros(prompt_length)
        prompt_attention[1:] = attns_per_head[-1][1:prompt_length]  # Take last position's attention

        attention_vector = prompt_attention.tolist()

        steps_attention.append(attention_vector)
    
    return steps_attention

def generate_stream(model, tokenizer, prompt, model_id):
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True,
        tokenize=False
    )
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    prompt_length = inputs.shape[1]
    streamer = TextIteratorStreamer(tokenizer)
    generation_output = None

    generation_kwargs = dict(
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        output_attentions=True,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=torch.ones_like(inputs),
    )
    
    def generate():
        nonlocal generation_output
        generation_output = model.generate(inputs, **generation_kwargs)
    
    thread = Thread(target=generate)
    thread.daemon = True
    thread.start()

    generated_text = ""
    for text in streamer:
        generated_text += text
        yield json.dumps({
            "model": model_id,
            "type": "token",
            "text": text,
        }) + "\n\n"
    
    thread.join()
    print(f"Model {model_id}: Generation complete. Text: {generated_text}")
    attention_scores = generation_output.attentions if generation_output is not None else None
    
    if attention_scores:
        steps_attention = process_attention_scores(attention_scores, prompt_length)
        print(f"Model {model_id}: attention scores: {len(steps_attention)} by {len(steps_attention[0])}")
        yield json.dumps({
            "model": model_id,
            "type": "attention",
            "prompt_tokens": tokenizer.convert_ids_to_tokens(inputs[0].tolist()),
            "attention_by_step": steps_attention
        }) + "\n\n"
    
    yield json.dumps({"model": model_id, "done": True}) + "\n\n"

@app.route("/attention")
def attention_view():
    prompt = request.args.get('prompt', '')
    print(f"Starting generation for prompt: {prompt[:50]}...")
    
    def generate():
        print("Initializing generators...")
        gen_a = generate_stream(model_a, model_a_tokenizer, prompt, 'a')
        gen_b = generate_stream(model_b, model_b_tokenizer, prompt, 'b')
        a_done = b_done = False
        
        while not (a_done and b_done):
            if not a_done:
                try:
                    data_a = next(gen_a)
                    print(f"Model A generated: {data_a[:50]}...")
                    if data_a:
                        yield f"data: {data_a}"
                except StopIteration:
                    print("Model A completed")
                    yield f"data: {json.dumps({'model': 'a', 'done': True})}\n\n"
                    a_done = True
                except Exception as e:
                    print(f"Error in Model A: {str(e)}")
                    yield f"data: {json.dumps({'model': 'a', 'error': str(e)})}\n\n"
                    a_done = True
            
            if not b_done:
                try:
                    data_b = next(gen_b)
                    print(f"Model B generated: {data_b[:50]}...")
                    if data_b:
                        yield f"data: {data_b}"
                except StopIteration:
                    print("Model B completed")
                    yield f"data: {json.dumps({'model': 'b', 'done': True})}\n\n"
                    b_done = True
                except Exception as e:
                    print(f"Error in Model B: {str(e)}")
                    yield f"data: {json.dumps({'model': 'b', 'error': str(e)})}\n\n"
                    b_done = True

    return Response(
        generate(),
        mimetype='text/event-stream'
    )

if __name__ == '__main__':
    init_models()
    app.run(debug=True, use_reloader=False)