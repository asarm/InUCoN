import pandas as pd
import numpy as np

from transformers import DistilBertModel, DistilBertTokenizer
import torch

import pickle
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def load_model(model_name='distilbert-base-uncased'):
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name)

    return tokenizer, model


def extract_features(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    features = last_hidden_states.mean(1).squeeze(0).numpy()

    return features


def save_description_vectors(save_path):
    tokenizer, model = load_model()
    textual_data = pd.read_csv("../cold_data/textual_information.csv")
    feature_vectors = {}
    for i in tqdm(range(len(textual_data))):
        row = textual_data.iloc[i]
        ticker_id = row["ticker_id"]
        desc = row["description"]

        feature_vector = extract_features(desc, tokenizer, model)
        feature_vectors[ticker_id] = feature_vector

    with open(save_path, 'wb') as f:
        pickle.dump(feature_vectors, f)


def load_json(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    return data


def prepare_textual_similarity_matrix(vectors_path="description_vectors.pkl"):
    text_vectors = load_json(vectors_path)
    similarity_matrix = cosine_similarity(list(text_vectors.values()))

    ''' Rescale Similarity Scores '''
    for id in range(len(similarity_matrix)):
        scaled_matrix = 2 * ((similarity_matrix[id] - min(similarity_matrix[id])) / (
                    max(similarity_matrix[id]) - min(similarity_matrix[id]))) - 1
        similarity_matrix[id] = scaled_matrix

    return similarity_matrix

# save_description_vectors(save_path='description_vectors.pkl')
# print("DONE")
# print(load_json("description_vectors.pkl"))