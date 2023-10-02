from typing import Optional, List, Dict, Union, Callable
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from IPython.display import SVG
from molfeat.trans.concat import FeatConcat
from molfeat.trans.fp import FPVecTransformer
import datamol as dm
from smilesfeature.constant import (
    ALL_REACTIVE_SITES,
    REACTION_CLASSES_TO_SMART_FRAGMENTS,
    DATAMOL_FEATURES,
)
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def add_molecule_from_smiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a new column "Molecule" to the dataframe by converting the "SMILES" column.

    Parameters:
    - df (pd.DataFrame): DataFrame with a "SMILES" column.

    Returns:
    - pd.DataFrame: DataFrame with an additional "Molecule" column.

    Example:
    >>> df = pd.DataFrame({"SMILES": ["C(=O)O", "CCO"]})
    >>> df_with_molecule = add_molecule_from_smiles(df)
    """

    df["Molecule"] = df["SMILES"].apply(Chem.MolFromSmiles)
    return df


def smiles_to_fp(smiles: str) -> np.ndarray:
    """
    Convert a SMILES string to a fingerprint array.

    Parameters:
    - smiles (str): A SMILES string.

    Returns:
    - np.ndarray: Fingerprint array of the molecule.

    Example:
    >>> smiles = "C(=O)O"
    >>> fp_array = smiles_to_fp(smiles)
    """

    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((1,), np.int32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def generate_3D_coordinates(smiles: str) -> Optional[np.ndarray]:
    """
    Generate 3D coordinates for a molecule from its SMILES string.

    Parameters:
    - smiles (str): A SMILES string.

    Returns:
    - np.ndarray: 3D coordinates of the molecule's atoms.
      Returns None if generating 3D coordinates fails.

    Example:
    >>> smiles = "C(=O)O"
    >>> coords = generate_3D_coordinates(smiles)
    """

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == -1:
        return None
    if mol.GetNumConformers() == 0:
        return None
    conformer = mol.GetConformer(0)
    positions = conformer.GetPositions()
    return positions


def smiles_to_image_array(smiles: str) -> np.ndarray:
    """
    Convert a SMILES string to an image array representation of its molecule.

    Parameters:
    - smiles (str): A SMILES string.

    Returns:
    - np.ndarray: Image array of the molecule.

    Example:
    >>> smiles = "C(=O)O"
    >>> image_arr = smiles_to_image_array(smiles)
    """

    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    image_array = np.array(img)
    return image_array

def _apply_pca(row, col_name):
    """
    Apply Principal Component Analysis (PCA) to a specific column in a DataFrame row.

    Args:
        row (pd.Series): The input DataFrame row.
        col_name (str): The name of the column to apply PCA to.

    Returns:
        pd.Series: A Series containing the PCA-transformed values.

    Example:
        >>> import pandas as pd
        >>> from sklearn.decomposition import PCA
        >>> data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        >>> df = pd.DataFrame(data)
        >>> pca = PCA(n_components=3)
        >>> sample_row = df.iloc[0]
        >>> transformed_values = _apply_pca(sample_row, col_name='feature1')
        >>> print(transformed_values)
    """
    pca = PCA(n_components=3)
    transformed = pca.fit_transform(row[col_name])
    new_col_names = [f'{col_name}_pc{i+1}' for i in range(3)]
    return pd.Series(transformed.mean(axis=0), index=new_col_names)


def apply_pca_to_dataframe(df, col_name):
    """
    Apply Principal Component Analysis (PCA) to a specific column in a DataFrame and add the PCA-transformed components as new columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
        col_name (str): The name of the column to apply PCA to.

    Returns:
        pd.DataFrame: A DataFrame with PCA-transformed columns added.

    Example:
        >>> import pandas as pd
        >>> from sklearn.decomposition import PCA
        >>> data = {'feature1': [1, 2, 3], 'feature2': [4, 5, 6]}
        >>> df = pd.DataFrame(data)
        >>> df_pca = apply_pca_to_dataframe(df, col_name='feature1')
        >>> print(df_pca.head())
    """
    new_columns = df.apply(lambda row: _apply_pca(row, col_name), axis=1)
    return pd.concat([df, new_columns], axis=1)


def perform_pca_on_mol2vec(df, n_components=3):
    """
    Perform Principal Component Analysis (PCA) on the 'Mol2Vec' column in a DataFrame and add the PCA-transformed components as new columns.

    Args:
        df (pd.DataFrame): The input DataFrame with a 'Mol2Vec' column.
        n_components (int, optional): The number of principal components to retain. Defaults to 3.

    Returns:
        pd.DataFrame: A DataFrame with PCA-transformed columns added.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from sklearn.decomposition import PCA
        >>> data = {'Mol2Vec': [np.random.rand(300) for _ in range(5)]}
        >>> df = pd.DataFrame(data)
        >>> df_pca = perform_pca_on_mol2vec(df, n_components=3)
        >>> print(df_pca.head())
    """
    if 'Mol2Vec' not in df.columns:
        raise ValueError("'Mol2Vec' column must be present in the DataFrame.")
    mol2vec_matrix = np.vstack(df['Mol2Vec'].values)
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(mol2vec_matrix)
    new_col_names = [f"Mol2Vec_pc{i+1}" for i in range(n_components)]
    df[new_col_names] = pd.DataFrame(pca_result, index=df.index)
    return df

    

def add_all_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add various molecular descriptors to a DataFrame based on the 'Molecule' column.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'Molecule' column.

    Returns:
    - pd.DataFrame: Updated DataFrame with new descriptor columns.

    Example:
    >>> df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    >>> df["Molecule"] = df["SMILES"].apply(Chem.MolFromSmiles)
    >>> df_with_descriptors = add_all_descriptors(df)
    """

    descriptors = [
        ("NumRings", Descriptors.RingCount),
        ("NumHDonors", Descriptors.NumHDonors),
        ("NumHAcceptors", Descriptors.NumHAcceptors),
        ("NumRotatableBonds", Descriptors.NumRotatableBonds),
        ("NumAromaticRings", Descriptors.NumAromaticRings),
        ("NumAliphaticRings", Descriptors.NumAliphaticRings),
    ]

    for name, descriptor in descriptors:
        df[name] = df["Molecule"].apply(descriptor)

    return df


def find_reactive_sites(mol: Chem.Mol) -> Dict[str, List[int]]:
    """
    Find reactive sites of the molecule.

    Parameters:
    - mol (Chem.Mol): RDKit molecule object.

    Returns:
    - Dict[str, List[int]]: Dictionary where keys are reactive site names and values
      are lists of atom indices corresponding to those reactive sites.

    Example:
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> reactive_sites = find_reactive_sites(mol)
    """

    reactive_sites = {}
    for site_name, pattern in ALL_REACTIVE_SITES.items():
        site_column = site_name.replace(" ", "_").replace("-", "_").replace(",", "")
        mol_pattern = Chem.MolFromSmarts(pattern)
        matches = mol.GetSubstructMatches(mol_pattern)
        reactive_sites[site_column] = [atom for match in matches for atom in match]

    return reactive_sites


def count_reactive_sites(mol: Chem.Mol) -> Dict[str, int]:
    """
    Count reactive sites of the molecule.

    Parameters:
    - mol (Chem.Mol): RDKit molecule object.

    Returns:
    - Dict[str, int]: Dictionary where keys are reactive site names and values
      are counts of those reactive sites in the molecule.

    Example:
    >>> mol = Chem.MolFromSmiles("CCO")
    >>> reactive_site_counts = count_reactive_sites(mol)
    """

    reactive_sites = find_reactive_sites(mol)
    return {key: len(value) for key, value in reactive_sites.items()}


def count_reaction_fragments(
    df: pd.DataFrame, REACTION_CLASSES_TO_SMILES_FRAGMENTS: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Count the occurrences of each fragment in the SMILES strings of the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'SMILES' column.
    - REACTION_CLASSES_TO_SMILES_FRAGMENTS (Dict[str, List[str]]): Dictionary where keys are reaction classes
      and values are lists of SMILES fragments.

    Returns:
    - pd.DataFrame: Updated DataFrame with count of each fragment in the SMILES strings.

    Example:
    >>> df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    >>> REACTION_CLASSES_TO_SMILES_FRAGMENTS = {"class1": ["CC"], "class2": ["CO"]}
    >>> updated_df = count_reaction_fragments(df, REACTION_CLASSES_TO_SMILES_FRAGMENTS)
    """

    for (
        reaction_class,
        smiles_fragments,
    ) in REACTION_CLASSES_TO_SMILES_FRAGMENTS.items():
        df[reaction_class] = df["SMILES"].apply(
            lambda s: sum(s.count(fragment) for fragment in smiles_fragments)
        )

    return df


def add_reactive_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add count and positions of reactive groups in the SMILES strings of the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'SMILES' column.

    Returns:
    - pd.DataFrame: Updated DataFrame with count and positions of each reactive group in the SMILES strings.

    Example:
    >>> df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    >>> updated_df = add_reactive_groups(df)
    """

    def count_and_positions(
        smiles: str, smarts_fragments: List[str]
    ) -> Dict[str, Union[int, str]]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"count": 0, "positions": ""}

        count = 0
        positions = []
        for fragment in smarts_fragments:
            patt = Chem.MolFromSmarts(fragment)
            matches = mol.GetSubstructMatches(patt)
            count += len(matches)
            positions.extend(matches)
        return {"count": count, "positions": str(positions)}

    for reaction_class, smarts_fragments in REACTION_CLASSES_TO_SMART_FRAGMENTS.items():
        df[f"{reaction_class}_count"] = 0
        df[f"{reaction_class}_positions"] = ""
        df_results = df["SMILES"].apply(
            count_and_positions, smarts_fragments=smarts_fragments
        )
        df[f"{reaction_class}_count"] = df_results.apply(lambda x: x["count"])
        df[f"{reaction_class}_positions"] = df_results.apply(lambda x: x["positions"])

    return df


def generate_descriptor_functions() -> Dict[str, Callable]:
    """
    Generate a dictionary of descriptor functions mapped by their names.

    Returns:
    - Dict[str, Callable]: Dictionary of descriptor names to their functions.

    Example:
    >>> descriptor_functions = generate_descriptor_functions()
    >>> descriptor_value = descriptor_functions["MolWt"]("CCO")
    """

    descriptor_fns = {
        desc: lambda smiles, d=desc: Descriptors.__dict__[d](Chem.MolFromSmiles(smiles))
        for desc, _ in Descriptors._descList
    }
    return descriptor_fns


def add_chem_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add chemical properties to the DataFrame using descriptor functions.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'SMILES' column.

    Returns:
    - pd.DataFrame: Updated DataFrame with added chemical properties.

    Example:
    >>> df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    >>> updated_df = add_chem_properties(df)
    """

    # Create a list to store the calculated chemical properties
    property_columns = []

    # Calculate chemical properties for each SMILES string
    for desc_name, desc_func in Descriptors.descList:
        properties = df["Molecule"].apply(desc_func)
        property_columns.append(pd.Series(properties, name=desc_name))

    # Concatenate the chemical properties columns to the original DataFrame
    df = pd.concat([df] + property_columns, axis=1)

    return df


def expand_reaction_sites(df: pd.DataFrame, max_pos: int = 30) -> pd.DataFrame:
    """
    Expand reaction sites into separate columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with a 'Reactive_Sites' column containing a dictionary.
    - max_pos (int, optional): Maximum position to be expanded, default is 30.

    Returns:
    - pd.DataFrame: Expanded DataFrame.

    Example:
    >>> df = pd.DataFrame({"Reactive_Sites": [{"A": [(0,1)], "B": [(2,3)]}]})
    >>> updated_df = expand_reaction_sites(df)
    """

    new_columns = []
    for reaction in df.loc[0, 'Reactive_Sites'].keys():
        for positions in df.loc[0, 'Reactive_Sites'][reaction]:
            if isinstance(positions, list):  # Check if positions is a list
                for pos in positions:
                    if pos <= max_pos:
                        new_columns.append(f"{reaction}_{pos}")
            
    df_new = pd.DataFrame(columns=new_columns)
    if len(new_columns) > 0:
        df_new.loc[0] = 0  # Initialize with zeros
    for reaction, positions in df.loc[0, 'Reactive_Sites'].items():
        for pos_tuple in positions:
            if isinstance(pos_tuple, list):  # Check if pos_tuple is a list
                pos = pos_tuple[0]  # Extract the position from the tuple
                if pos <= max_pos:
                    col_name = f"{reaction}_{pos}"
                    df_new.at[0, col_name] = 1
                
    df = pd.concat([df, df_new], axis=1)
    df.fillna(0, inplace=True)
    return df





def generate_chemical_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate chemical properties for each molecule in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with a column named 'SMILES'.

    Returns:
    - pd.DataFrame: Updated DataFrame with chemical properties.

    Example:
    >>> df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    >>> updated_df = generate_chemical_properties(df)
    """
    df["Mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    df["Num_H_Acceptors"] = df["Mol"].apply(Descriptors.NumHAcceptors)
    df["Num_H_Donors"] = df["Mol"].apply(Descriptors.NumHDonors)
    df["Num_RotatableBonds"] = df["Mol"].apply(Descriptors.NumRotatableBonds)
    return df.drop(columns=["Mol"])


def interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate missing values in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame with potential missing values.

    Returns:
    - pd.DataFrame: DataFrame with interpolated values.

    Example:
    >>> df = pd.DataFrame({"A": [1, np.nan, 3]})
    >>> updated_df = interpolate_missing_values(df)
    """
    if df.isnull().any().any():
        return df.interpolate()
    return df


def preprocess_mol(smiles: str) -> str:
    """
    Preprocess a molecule represented as SMILES using datamol.

    Parameters:
    - smiles (str): Input SMILES string.

    Returns:
    - str: Preprocessed SMILES string.
    """
    mol = dm.to_mol(smiles, ordered=True)
    mol = dm.fix_mol(mol)
    mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
    mol = dm.standardize_mol(
        mol,
        disconnect_metals=False,
        normalize=True,
        reionize=True,
        uncharge=False,
        stereo=True,
    )
    return dm.to_smiles(mol)


def extract_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract additional features for molecules using datamol.

    Parameters:
    - df (pd.DataFrame): DataFrame with a column named 'SMILES'.

    Returns:
    - pd.DataFrame: Updated DataFrame with extracted features.

    Example:
    >>> df = pd.DataFrame({"SMILES": ["CCO", "CCN"]})
    >>> updated_df = extract_extra_features(df)
    """
    df["Standard_Smiles"] = df["SMILES"].apply(preprocess_mol)
    fps = [FPVecTransformer(fp, dtype=np.float64, n_jobs=-1) for fp in DATAMOL_FEATURES]
    featurizer = FeatConcat(fps, dtype=np.float64)
    descriptors = featurizer(df["Standard_Smiles"].to_list())

    new_columns = [f"feature_{fp}" for fp in DATAMOL_FEATURES]
    for i, col in enumerate(new_columns):
        df[col] = descriptors[:, i]

    return df


def add_descriptors_to_df(df: pd.DataFrame):
    # Convert SMILES strings to mol objects and store them in a new column 'mol'
    df["mol"] = df["SMILES"].apply(Chem.MolFromSmiles)
    
    # Compute all descriptors for all molecules in one go
    df['dm_descriptor_dict'] = df["mol"].apply(dm.descriptors.compute_many_descriptors)
    
    # Create new columns for each descriptor and set them to None
    for descriptor_name in df['dm_descriptor_dict'].iloc[0].keys():
        df[descriptor_name] = None
    
    # Iterate over rows and set descriptor values
    for index, row in df.iterrows():
        dm_descriptor_dict = row['dm_descriptor_dict']
        for descriptor_name, value in dm_descriptor_dict.items():
            df.at[index, descriptor_name] = value
    
    # Drop the 'mol' and 'dm_descriptor_dict' columns if they are no longer needed
    # df.drop(['mol', 'dm_descriptor_dict'], axis=1, inplace=True)
    
    return df
