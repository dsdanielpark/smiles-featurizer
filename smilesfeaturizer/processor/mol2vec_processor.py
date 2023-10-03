import numpy as np
from typing import List
import datamol as dm
import pandas as pd
from rdkit import Chem
from gensim.models import Word2Vec
from mol2vec.features import mol2alt_sentence
from molfeat.calc import RDKitDescriptors2D
from molfeat.trans import MoleculeTransformer, FPVecTransformer


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


def calculate_and_add_ecfp_fingerprints(
    df: pd.DataFrame, smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    Calculate ECFP fingerprints from the 'SMILES' column and add them as new columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'SMILES' column.
        smiles_col (str): The name of the column containing SMILES strings. Default is 'SMILES'.

    Returns:
        pd.DataFrame: The input DataFrame with added ECFP fingerprints as new columns.
    """
    # Extract SMILES strings from the specified column
    data = df[smiles_col].values

    # Initialize FPVecTransformer for 'ecfp:4'
    ecfp4 = FPVecTransformer("ecfp:4", dtype=np.float32)

    # Calculate ECFP fingerprints for each SMILES string
    ecfp4_fingerprints = ecfp4(data)

    # Define column names for the ECFP fingerprints
    ecfp4_columns = [f"ecfp4_{i}" for i in range(ecfp4_fingerprints.shape[1])]

    # Create a DataFrame from the calculated ECFP fingerprints
    ecfp4_df = pd.DataFrame(ecfp4_fingerprints, columns=ecfp4_columns)

    # Concatenate the original DataFrame and the ECFP fingerprint DataFrame
    result_df = pd.concat([df, ecfp4_df], axis=1)

    return result_df


def calculate_and_add_rdkit2d_descriptors(
    df: pd.DataFrame, smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    Calculate RDKit 2D descriptors from the 'SMILES' column and add them as new columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'SMILES' column.
        smiles_col (str): The name of the column containing SMILES strings. Default is 'SMILES'.

    Returns:
        pd.DataFrame: The input DataFrame with added RDKit 2D descriptors as new columns.
    """
    # Extract SMILES strings from the specified column
    data = df[smiles_col].values

    # Initialize the RDKit 2D descriptor calculator
    calc = RDKitDescriptors2D(replace_nan=True)

    # Wrap the calculator in a transformer instance
    featurizer = MoleculeTransformer(calc, dtype=np.float64)

    # Calculate RDKit 2D descriptors for each SMILES string
    with dm.without_rdkit_log():
        feats = featurizer(data)

    # Define column names for the RDKit 2D descriptors
    feature_names = [f"rdkit2d_{i}" for i in range(feats.shape[1])]

    # Create a DataFrame from the calculated descriptors
    features_df = pd.DataFrame(feats, columns=feature_names)

    # Concatenate the original DataFrame and the features DataFrame
    result_df = pd.concat([df, features_df], axis=1)

    return result_df


def calculate_and_add_maccs_fingerprints(
    df: pd.DataFrame, smiles_col: str = "SMILES"
) -> pd.DataFrame:
    """
    Calculate MACCS fingerprints from the 'SMILES' column and add them as new columns to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'SMILES' column.
        smiles_col (str, optional): The name of the column containing SMILES strings. Default is 'SMILES'.

    Returns:
        pd.DataFrame: The input DataFrame with added MACCS fingerprints as new columns.
    """
    # Extract SMILES strings from the specified column
    data = df[smiles_col].values

    # Initialize FPVecTransformer for 'maccs'
    maccs = FPVecTransformer("maccs", dtype=np.float32)

    # Calculate MACCS fingerprints for each SMILES string
    maccs_fingerprints = maccs(data)

    # Define column names for the MACCS fingerprints
    maccs_columns = [f"maccs_{i}" for i in range(maccs_fingerprints.shape[1])]

    # Create a DataFrame from the calculated MACCS fingerprints
    maccs_df = pd.DataFrame(maccs_fingerprints, columns=maccs_columns)

    # Concatenate the original DataFrame and the MACCS fingerprint DataFrame
    result_df = pd.concat([df, maccs_df], axis=1)

    return result_df
