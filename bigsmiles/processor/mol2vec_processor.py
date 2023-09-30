import numpy as np
from rdkit import Chem
from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence
from typing import List


def sentences2vec(sentences: List[str], model, unseen=None) -> np.ndarray:
    """
    Convert a list of sentences into their vector representations using a given Word2Vec model.

    Parameters:
    - sentences (List[str]): A list of sentences where each sentence is a list of words.
    - model: Pre-trained Word2Vec model.
    - unseen (str, optional): Token to use for unseen words. If None, unseen words are ignored.

    Returns:
    - np.ndarray: An array where each row is the vector representation of a sentence.

    Example:
    >>> model = Word2Vec(sentences)  # assuming sentences is predefined
    >>> sentences = [['cat', 'sat'], ['dog', 'barked']]
    >>> vecs = sentences2vec(sentences, model)
    """

    keys = set(model.wv.key_to_index)
    vec = []

    unseen_vec = None
    if unseen:
        unseen_vec = model.wv[unseen]

    for sentence in sentences:
        sentence_vec = np.zeros(model.wv.vector_size)

        for word in sentence:
            if word in keys:
                sentence_vec += model.wv[word]
            elif unseen_vec is not None:
                sentence_vec += unseen_vec

        vec.append(sentence_vec)

    return np.array(vec)


def mol2vec_feature(smiles: str, mol2vec_model: Word2Vec) -> np.ndarray:
    """
    Convert a SMILES string into its molecular vector representation using a given Word2Vec model.

    Parameters:
    - smiles (str): A SMILES string.
    - mol2vec_model: Pre-trained mol2vec Word2Vec model.

    Returns:
    - np.ndarray: Vector representation of the molecule. Returns None if conversion fails.

    Example:
    >>> mol2vec_model = Word2Vec(sentences)  # assuming sentences is predefined
    >>> smiles = "C(C(=O)O)N"
    >>> vec = mol2vec_feature(smiles, mol2vec_model)
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("Failed to generate molecule from SMILES string.")
            return None

        sentence = mol2alt_sentence(mol, 1)
        vector = sentences2vec([sentence], mol2vec_model, unseen="UNK")
        return vector
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
