import modal

image = modal.Image.debian_slim().pip_install(
    "bitsandbytes",
    "transformers[torch]",
    "flask"
)
app = modal.App(name="has-simple-web-endpoint", image=image)

mount = modal.Mount.from_local_dir(".", remote_path="/root")


@app.function(gpu="any", image=image, mounts=[mount])
@modal.wsgi_app()
def flask_app():
    from flask import Flask, Response, request, send_from_directory
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer, BitsAndBytesConfig
    from threading import Thread
    import json
    import torch
    import os

    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(current_dir, 'static')


    RANDOM_SEED = 42
    torch.manual_seed(RANDOM_SEED)

    web_app = Flask(__name__, static_folder=static_dir)

    device = "cuda"
    max_new_tokens = 350

    model_id = "HuggingFaceTB/SmolLM2-360M-Instruct"

    def load_model(model_id, quantize=False):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if quantize:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
            )
        model.eval()
        return tokenizer, model
    
    model_a_tokenizer, model_a = load_model(model_id)
    model_b_tokenizer, model_b = load_model(model_id, quantize=True)

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
        with torch.inference_mode():
                
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
                do_sample=True
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
            attention_scores = generation_output.attentions if generation_output is not None else None
            
            if attention_scores:
                steps_attention = process_attention_scores(attention_scores, prompt_length)
                yield json.dumps({
                    "model": model_id,
                    "type": "attention",
                    "prompt_tokens": tokenizer.convert_ids_to_tokens(inputs[0].tolist()),
                    "attention_by_step": steps_attention
                }) + "\n\n"
            
            yield json.dumps({"model": model_id, "done": True}) + "\n\n"

    @web_app.route('/')
    def index():
        return send_from_directory(static_dir, 'index.html')

    @web_app.route("/attention")
    def attention_view():
        prompt = request.args.get('prompt', '')
        
        def generate():
            gen_a = generate_stream(model_a, model_a_tokenizer, prompt, 'a')
            gen_b = generate_stream(model_b, model_b_tokenizer, prompt, 'b')
            a_done = b_done = False
            
            while not (a_done and b_done):
                if not a_done:
                    try:
                        data_a = next(gen_a)
                        if data_a:
                            yield f"data: {data_a}"
                    except StopIteration:
                        yield f"data: {json.dumps({'model': 'a', 'done': True})}\n\n"
                        a_done = True
                    except Exception as e:
                        yield f"data: {json.dumps({'model': 'a', 'error': str(e)})}\n\n"
                        a_done = True
                
                if not b_done:
                    try:
                        data_b = next(gen_b)
                        if data_b:
                            yield f"data: {data_b}"
                    except StopIteration:
                        yield f"data: {json.dumps({'model': 'b', 'done': True})}\n\n"
                        b_done = True
                    except Exception as e:
                        yield f"data: {json.dumps({'model': 'b', 'error': str(e)})}\n\n"
                        b_done = True

        return Response(
            generate(),
            mimetype='text/event-stream'
        )
    
    return web_app