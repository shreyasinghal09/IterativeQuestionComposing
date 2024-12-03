from os.path import join
import os
import json
from typing import List, Optional, Union

# import pandas as pd
# import fire
import torch
# from vllm import LLM, SamplingParams, RequestOutput
from tqdm import tqdm

import ttt

from utils.normalize_answer import compare_modelanswer_with_answer, extract_math_answer
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def get_question_prompt(question):
    prefix = 'Please solve the following problem and put your answer at the end with "The answer is: ".\n\n'
    return prefix + question

# Initialize embedding model
embedding_model_name = "math-similarity/Bert-MLM_arXiv-MP-class_zbMath"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={"trust_remote_code": True}
)

# Initialize Chroma
persist_directory = "/data/user_data/amittur/MMIQC_ChromaDB"
vectorstore = Chroma(
    embedding_function=embeddings_model,
    persist_directory=persist_directory,
)

def get_top_k_similar_questions(query, top_k):
    results = vectorstore.similarity_search(query, k=top_k)
    return [
        {
            "question": result.page_content,
            "solution": result.metadata["answer"]
        }
        for result in results
    ]

def eval_MATH_ttt(
    test_file,
    output_root: str = 'output',
    output_name: str = 'default',
    ):
    os.makedirs(output_root, exist_ok=True)
    output_fn = join(output_root, f'{output_name}.jsonl')
    with open(test_file, 'r') as f:
        # has answer, problem and solution fields
        data_points = [json.loads(line) for line in f]
    problems = [dp['problem'] for dp in data_points]
    answers = [dp['answer'] for dp in data_points]
    solutions = [dp['solution'] for dp in data_points]

    num_correct, current_total = 0, 0
    for problem, answer, solution in tqdm(zip(problems, answers, solutions)):
        model_solution = generate_MATH_solution(problem)
        model_answer = extract_math_answer(model_solution)
        correct = compare_modelanswer_with_answer(answer, model_answer)
        current_total += 1
        num_correct += correct
        data_point = {
                'correct': correct, 'answer': answer, 'model_answer': model_answer,
                'problem': problem, 'solution': solution, 'model_solution': model_solution 
            }
        with open(output_fn, 'a') as f:
            f.write(json.dumps(data_point)+'\n')
    message = f'{num_correct/current_total:.4f}, {num_correct}/{current_total}, {output_fn}'
    print(message)
        
    

def generate_MATH_solution(
    problem,
    ):
    query = get_question_prompt(problem)
    similar_questions = get_top_k_similar_questions(query, top_k=2000)
    solution = ttt.ttt_predict(similar_questions, query)
    # finetuned_model = get_tuned_model(similar_questions)
    # sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, stop=stop)
    # tokenizer = finetuned_model.get_tokenizer()
    # if not tokenizer.eos_token_id:
    #     try:
    #         tokenizer.eos_token_id = tokenizer.eod_id
    #         print('Now setting eos_token_id to eod_id for Qwen models')
    #     except Exception as e:
    #         raise(f'No "eos_token_id" or "eod_id" for the tokenizer. Please specify one.')
    # prompts = [get_question_prompt(problem)]
    # output = finetuned_model.generate(prompts, sampling_params)[0] # type: RequestOutput
    # model_solution = output.outputs[0].text
    # return model_solution
    return solution

if __name__ == '__main__':
    # Take in test file as an argument and call eval_MATH_ttt
    eval_MATH_ttt('datasets/MATH-test_small.jsonl', output_root='output', output_name='ttt_small')