import os
import pickle
import pandas as pd
import numpy as np
from gensim.models import Word2Vec


class ModelLoaderService:
    # Cached Artifacts Loader
    # هنا نحمل ملفات الكاش الجاهزة بدل إعادة بناء الداتا والنماذج عند كل تشغيل.
    def __init__(self, save_dir="saved_files"):
        self.save_dir = save_dir
        self.data = {}

    def load_pickle(self, file_name):
        path = os.path.join(self.save_dir, file_name)
        with open(path, "rb") as file:
            return pickle.load(file)

    def load_csv(self, file_name):
        path = os.path.join(self.save_dir, file_name)
        return pd.read_csv(path)

    def load_numpy(self, file_name):
        path = os.path.join(self.save_dir, file_name)
        return np.load(path)

    def load_word2vec(self, file_name):
        path = os.path.join(self.save_dir, file_name)
        return Word2Vec.load(path)

    def load_all(self):
        # Service Oriented Architecture
        # هنا نجمع كل ملفات الداتا والنماذج لتستخدمها خدمات البحث بشكل منفصل.
        self.data["work1_df"] = self.load_csv("work_dataset1.csv")
        self.data["work2_df"] = self.load_csv("work_dataset2.csv")

        self.data["tfidf_vectorizer_1"] = self.load_pickle("tfidf_vectorizer_1.pkl")
        self.data["tfidf_vectorizer_2"] = self.load_pickle("tfidf_vectorizer_2.pkl")

        self.data["tfidf_matrix_1"] = self.load_pickle("tfidf_matrix_1.pkl")
        self.data["tfidf_matrix_2"] = self.load_pickle("tfidf_matrix_2.pkl")

        self.data["word2vec_model_1"] = self.load_word2vec("word2vec_model_dataset1.model")
        self.data["word2vec_model_2"] = self.load_word2vec("word2vec_model_dataset2.model")

        self.data["word2vec_matrix_1"] = self.load_numpy("word2vec_matrix_1.npy")
        self.data["word2vec_matrix_2"] = self.load_numpy("word2vec_matrix_2.npy")

        self.data["tokenized_corpus_1"] = self.load_pickle("tokenized_corpus_1.pkl")
        self.data["tokenized_corpus_2"] = self.load_pickle("tokenized_corpus_2.pkl")

        self.data["inverted_index_1"] = self.load_pickle("inverted_index_1.pkl")
        self.data["inverted_index_2"] = self.load_pickle("inverted_index_2.pkl")

        return self.data
