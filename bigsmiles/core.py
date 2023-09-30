import re
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from sklearn.cluster import DBSCAN
from bigsmiles.processor.mol2vec_processor import sentences2vec, mol2vec_feature
from bigsmiles.processor.smiles_processor import add_molecule_from_smiles, smiles_to_fp, generate_3D_coordinates, smiles_to_image_array, perform_pca_on_column, add_all_descriptors, find_reactive_sites, count_reactive_sites, count_reaction_fragments, add_reactive_groups, generate_descriptor_functions, add_chem_properties, expand_reaction_sites, generate_chemical_properties, interpolate_missing_values, extract_extra_features
from bigsmiles.constant import REACTION_CLASSES_TO_SMILES_FRAGMENTS
from gensim.models import Word2Vec


def generate(df, method="simple"):
    mol2vec_model = Word2Vec.load("./mol2vel/model_300dim.pkl")
    # Preprocessing steps
    df = interpolate_missing_values(df)
    df['Mol'] = df['SMILES'].apply(Chem.MolFromSmiles)
    df = generate_chemical_properties(df)

    # Feature Engineering steps
    df['fingerprint'] = df['SMILES'].apply(smiles_to_fp)
    df[['3D_Coordinates', 'Mol2Vec', 'image_array']] = df['SMILES'].apply(
        lambda x: pd.Series({
            '3D_Coordinates': generate_3D_coordinates(x),
            'Mol2Vec': mol2vec_feature(x, mol2vec_model).reshape(-1),
            'image_array': smiles_to_image_array(x)
        })
    )
    df = perform_pca_on_column(df, '3D_Coordinates')
    df = perform_pca_on_column(df, "Mol2Vec")
    df["Mol2Vec_mean"] = df["Mol2Vec"].apply(np.mean)
    df["Mol2Vec_std"] = df["Mol2Vec"].apply(np.std)

    # Chemical Property steps
    df = add_reactive_groups(df)
    df = add_all_descriptors(df)
    df['Reactive_Sites'] = df['Mol'].apply(find_reactive_sites)
    df['Reactive_Sites_Count'] = df['Mol'].apply(count_reactive_sites)
    for site in ['Nitro_Reduction', 'Carbonyl_Reduction', 'Dehydration', 'Amides', 'Oxidations', 'Epoxidation', 'Oxidative_Dealkylation']:
        df[site] = df['Reactive_Sites_Count'].apply(lambda x: x.get(site, 0))

    # Clustering
    mol2vec_matrix = np.array(df['Mol2Vec'].tolist())
    dbscan = DBSCAN(eps=1, min_samples=2)
    df['Cluster'] = dbscan.fit_predict(mol2vec_matrix)

    # Reaction steps
    df = count_reaction_fragments(df, REACTION_CLASSES_TO_SMILES_FRAGMENTS)
    df = add_chem_properties(df)

    # Clean column names
    df.columns = [re.sub('[^0-9a-zA-Z]+', '_', col) for col in df.columns]

    # Reaction Site expansion
    df = expand_reaction_sites(df)

    # Optional additional features
    if method == "more":
        df = extract_extra_features(df)
        df.drop(columns=['clogp', 'mw', 'tpsa'], inplace=True)

    return df

if __name__ =="__main__":
    df = pd.read_csv("./data/big_smiles.csv")
    df_test = df[:10]
    df_test = generate(df_test, method="simple")
    df_test.to_csv("all.csv")

