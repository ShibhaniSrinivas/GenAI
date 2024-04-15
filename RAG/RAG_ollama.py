import ollama 
import time
import os
import json
import numpy as np
from numpy.linalg import norm
from striprtf.striprtf import rtf_to_text
from pathlib import Path


def parse_file(filename):
    with open(filename, encoding="utf-8-sig") as f:
        paragraphs, buffer = [], []
        for line in f.readlines():
            line = line.strip()
            if line:
                buffer.append(line)
            elif len(buffer):
                paragraphs.append((" ").join(buffer))
                buffer = []
        if len(buffer):
            paragraphs.append((" ").join(buffer))
    return paragraphs


def save_embeddings(filename, embeddings):
    if not os.path.exists("embeddings"):
        os.makedirs("embeddings")
    with open(f"embeddings/{filename}.json", "w") as f:
        json.dump(embeddings, f)


def load_embeddings(filename):
    if not os.path.exists(f"embeddings/{filename}.json"):
        return False
    with open(f"embeddings/{filename}.json", "r") as f:
        return json.load(f)
    

def get_embeddings(filename, modelname, chunks):
    if (embeddings := load_embeddings(filename)) is not False:
        return embeddings
    embeddings = [
        ollama.embeddings(model=modelname, prompt=chunk)["embedding"]
        for chunk in chunks
    ]
    save_embeddings(filename, embeddings)
    return embeddings


def find_most_similar(key, value):
    key_norm = norm(key)
    similarity_scores = [
        np.dot(key, item) / (key_norm * norm(item)) for item in value
    ]
    return sorted(zip(similarity_scores, range(len(value))), reverse=True)


def main():
    system_prompt =  """You are a helpful reading assistant who answers questions 
        based on snippets of text provided in context. Answer only using the context provided, 
        being as concise as possible. If you're unsure, just say that you don't know.
        Context:
    """
    rtf_filename = "/Users/shibhanisrinivas/Desktop/Interviews_USA/InterviewPrep/GenAI/Datasets/PeterPan.rtf"
    # filename = "GenAI/Datasets/PeterPan.rtf"
    txt_filename = rtf_filename.split("/")[-1].split(".")[0]+".txt"
    convert_rtf_to_txt(rtf_filename, txt_filename)

    paragraphs = parse_file(txt_filename)

    embeddings = get_embeddings(txt_filename, "nomic-embed-text", paragraphs)

    prompt = input("What do you want to know? ->")
    prompt_embedding = ollama.embeddings(model="nomic-embed-text", prompt=prompt)["embedding"]
    most_similar_chunks = find_most_similar(prompt_embedding, embeddings)[:5]

    response = ollama.chat(
        model="mistral",
        messages=[
            {
                "role": "system",
                "content": system_prompt
                + "\n".join(paragraphs[item[1]] for item in most_similar_chunks),
            },
            {"role": "user", "content": prompt},
        ],
    )
    print("\n\n")
    print(response["message"]["content"])


def convert_rtf_to_txt(rtf_filepath, txt_filepath):
    with open(rtf_filepath, 'r') as f:
        rtf_content = f.read()

    text_content = rtf_to_text(rtf_content)

    with open(txt_filepath, 'w') as f:
        f.write(text_content)



if __name__ == "__main__":
    base_directory = Path.cwd()
    current_directory = base_directory / 'RAG'
    os.chdir(current_directory)
    main()



