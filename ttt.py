from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)

training_args = SFTConfig(
    output_dir='checkpoints',
    packing=True,
    num_train_epochs=1,              # number of training epochs
    per_device_train_batch_size=2,   # batch size for training
    gradient_accumulation_steps=1,   # accumulate gradients over 8 steps
    save_total_limit=1,               # limit the total amount of checkpoints
    logging_steps=50,
)

MODEL_NAME = 'mistralai/Mistral-7B-v0.1'

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    cache_dir='/data/user_data/amittur',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# tokenizer.pad_token = '<unk>'
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = 'right'

# data = [{'question': 'Please solve the following problem and put your answer at the end with "The answer is: ".\n\nWhat is $\\frac{2}{5}$ divided by 3?\n\n',
#   'solution': 'To divide $\\frac{2}{5}$ by 3, we can rewrite it as $\\frac{2}{5}\\div\\frac{3}{1}$.\nDividing by a fraction is the same as multiplying by its reciprocal, so we have $\\frac{2}{5}\\cdot\\frac{1}{3}$.\nMultiplying the numerators and denominators, we get $\\frac{2\\cdot1}{5\\cdot3}$.\nSimplifying, we have $\\frac{2}{15}$.\nTherefore, $\\frac{2}{5}$ divided by 3 is $\\boxed{\\frac{2}{15}}$.\nThe answer is: \\frac{2}{15}'},
#  {'question': 'Please solve the following problem and put your answer at the end with "The answer is: ".\n\nWhat is $\\frac{2}{5}$ divided by 3?\n\n',
#   'solution': "To divide a fraction by a whole number, you can multiply the denominator of the fraction by the whole number. Alternatively, you can think of it as multiplying the fraction by the reciprocal of the whole number. Let's use the second method:\n\n$$ \\frac{2}{5} \\div 3 = \\frac{2}{5} \\times \\frac{1}{3} $$\n\nNow, multiply the numerators and the denominators:\n\n$$ \\frac{2 \\times 1}{5 \\times 3} = \\frac{2}{15} $$\n\nTherefore, $\\frac{2}{5}$ divided by 3 is $\\boxed{\\frac{2}{15}}$. The answer is: \\frac{2}{15}"},
#  {'question': 'Please solve the following problem and put your answer at the end with "The answer is: ".\n\nWhat is $\\frac{2}{5}$ divided by 3?\n\n',
#   'solution': 'To divide a fraction by a whole number, we need to multiply the numerator of the fraction by the reciprocal of the whole number.\n\nThe reciprocal of 3 is $\\frac{1}{3}$.\n\nTherefore, $\\frac{2}{5}$ divided by 3 is given by:\n\n$$\\frac{2}{5} \\div 3 = \\frac{2}{5} \\times \\frac{1}{3}$$\n\nMultiplying the numerators and denominators, we get:\n\n$$\\frac{2 \\times 1}{5 \\times 3} = \\frac{2}{15}$$\n\nHence, $\\frac{2}{5}$ divided by 3 is $\\boxed{\\frac{2}{15}}$. The answer is: \\frac{2}{15}'}]

def ttt_data_gen(data):
    def curry():
        for d in data:
            # yield {
            #     'prompt': f'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n{d["question"]}\n\n',
            #     'completion': d['solution']
            # }
            yield {
                'text': f"{d['question']}\n\n{d['solution']}"
            }
    return curry

def ttt_predict(data, question):
    global model, tokenizer, lora_config, training_args

    dataset = Dataset.from_generator(ttt_data_gen(data))

    trainer = SFTTrainer(
        model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
    )

    # trainer.processing_class.padding_side = 'right'

    trainer.train()


    # print(data)
    # print(data[0])
    # print(question)
    in_token_ids = tokenizer(question, return_tensors='pt')

    out_token_ids = trainer.model.generate(
        **in_token_ids.to(trainer.model.device),
        max_new_tokens=500,
        do_sample=False,
        temperature=None,
        top_p=None,
    ).detach().cpu()

    # print(in_token_ids)
    # print(out_token_ids)

    # print(tokenizer.decode(out_token_ids[0]))
    completion = tokenizer.decode(out_token_ids[0][len(in_token_ids['input_ids'][0]):], skip_special_tokens=True)
    # print(completion)

    model = trainer.model.unload()

    return completion