from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = Flask(__name__)

# Load pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the locally downloaded "llama2" model
model_path = "/mnt/d/Model/llama-2-7b"  # Replace this with the actual path to your "llama2" model
model = GPT2LMHeadModel.from_pretrained(model_path, pad_token_id=tokenizer.eos_token_id)

@app.route('/api/generate', methods=['POST'])
def generate_response():
    # Check if the request has JSON data
    if not request.json:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    # Extract model and prompt from the JSON data
    model_name = request.json.get('model')
    prompt = request.json.get('prompt')

    # Validate that both model and prompt are provided
    if not model_name or not prompt:
        return jsonify({'error': 'Both model and prompt are required'}), 400
    
    # Load the specified model
    if model_name != "llama2":
        return jsonify({'error': f'Model "{model_name}" is not supported'}), 400
    
    # Tokenize the prompt and generate text
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # Decode and return the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({'response': generated_text}), 200

if __name__ == '__main__':
    app.run(port=11434, debug=True)
