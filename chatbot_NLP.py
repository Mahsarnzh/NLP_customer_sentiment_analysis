from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)


# Function to generate a filtered response
def generate_filtered_response(prompt, max_length=50, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, device=model.device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        attention_mask=attention_mask,
        no_repeat_ngram_size=2,  # Filter out repeated n-grams
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


# Chatbot loop
print("Chatbot: Hello! How can I assist you today? (Type 'exit' to end the conversation)")

previous_response = ""

while True:
    user_input = input("You: ")

    if user_input.lower() == 'exit':
        print("Chatbot: Goodbye!")
        break

    # Generate and print the chatbot's response
    response = generate_filtered_response(user_input, max_length=50, num_return_sequences=1)

    # Check if the response is different from the previous one
    if response != previous_response:
        print("Chatbot:", response)
        previous_response = response
    else:
        print("Chatbot: I'm not sure what you mean. Can you please provide more details?")
