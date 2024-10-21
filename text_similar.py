import torch

from sentence_transformers import SentenceTransformer, util

#sys.stderr = open(os.devnull, 'w')

model = SentenceTransformer('all-MiniLM-L6-v2')

refrence_text = input("Enter refrence text: ")

input_texts = [
    "change the fan LED to red",
    "change the fan LED to orange",
    "red",
    "orange",
    "green",
    "blue",
]

similarity = util.cos_sim(
    model.encode(refrence_text, convert_to_tensor=True),
    model.encode(input_texts, convert_to_tensor=True)
)[0].cpu().tolist()

print(similarity)
