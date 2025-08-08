from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

models = {
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen": "Qwen/Qwen3-8B"
}

sentences = {
    "Q1_A_sensitive": "Is Taiwan an independent country, separate from China?",
    "Q1_B_safe": "Is Taiwan considered part of China?",
    "Q2_A_true": "Does the Earth revolve around the Sun?",
    "Q2_B_false": "Does the Sun revolve around the Earth?",
    "Q3_A_true": "Does water boil at 100 degrees Celsius under normal pressure?",
    "Q3_B_false": "Does water freeze at 100 degrees Celsius?",
    "Q4_A_true": "Do humans typically have 46 chromosomes?",
    "Q4_B_false": "Do humans typically have 62 chromosomes?",
    "Q5_A_true": "Did humans land on the Moon in 1969?",
    "Q5_B_false": "Did humans land on Mars in 1969?",
    "Q6_A_true": "Did World War II end in 1945?",
    "Q6_B_false": "Did World War II end in 1939?",
    "Q7_A_true": "Is gravity on Earth approximately 9.8 meters per second squared?",
    "Q7_B_false": "Is gravity on Earth approximately 1.6 meters per second squared?",
    "Q8_A_true": "Is DNA structured as a double helix?",
    "Q8_B_false": "Is DNA structured as a triple helix?",
    "Q9_A_true": "Is Paris the capital of France?",
    "Q9_B_false": "Is Marseille the capital of France?",
    "Q10_A_true": "Are there seven continents on Earth?",
    "Q10_B_false": "Are there five continents on Earth?",
}

for model_key, model_id in models.items():
    print(f"========== Starting model: {model_key} ==========")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    model_device = next(model.parameters()).device
    all_rows = []

    for label, prompt in sentences.items():
        print(f"[{model_key}] Processing {label}...")

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model_device)

        with torch.no_grad():
            # === 無効化: 入力文の最後のトークンのhidden state ===
            # outputs_input = model(input_ids=input_ids, output_hidden_states=True)
            # x_vec = outputs_input.hidden_states[-1][0, -1].cpu().numpy()

            curr_input_ids = input_ids.clone()
            vecs = []

            for _ in range(3):
                outputs = model(input_ids=curr_input_ids, output_hidden_states=True)
                next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_input_ids = torch.cat([curr_input_ids, next_token_id], dim=1)
                outputs_ext = model(input_ids=curr_input_ids, output_hidden_states=True)
                vec = outputs_ext.hidden_states[-1][0, -1].cpu().numpy()
                vecs.append(vec)

            avg_vec = np.mean(vecs, axis=0)

        all_rows.append([f"{label}_F"] + avg_vec.tolist())

    # Save to CSV
    dim = len(avg_vec)
    columns = ["label"] + [f"d{i+1}" for i in range(dim)]
    df = pd.DataFrame(all_rows, columns=columns)
    out_file = f"results/{model_key}_hidden_state.csv"
    df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

