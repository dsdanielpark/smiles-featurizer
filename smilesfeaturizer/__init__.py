# Copyright 2023 Minwoo Park, Apache 2.0 License.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# # limitations under the License.
from smilesfeaturizer.processor.mol2vec_processor import (
    calculate_and_add_ecfp_fingerprints,
    calculate_and_add_maccs_fingerprints,
    calculate_and_add_rdkit2d_descriptors,
    mol2vec_feature,
)
from smilesfeaturizer.processor.smiles_processor import (
    add_molecule_from_smiles,
    smiles_to_fp,
    generate_3D_coordinates,
    smiles_to_image_array,
    add_all_descriptors,
    find_reactive_sites,
    count_reactive_sites,
    count_reaction_fragments,
    add_reactive_groups,
    generate_descriptor_functions,
    add_chem_properties,
    expand_reaction_sites,
    generate_chemical_properties,
    interpolate_missing_values,
    perform_pca_on_mol2vec,
    apply_pca_to_dataframe,
    extract_extra_features,
    add_descriptors_to_df,
)
from smilesfeaturizer.core import generate_smiles_feature
from smilesfeaturizer.analysis.dash import create_inline_dash_dashboard
from smilesfeaturizer.analysis.plotter import draw_corr, df_scatter_plot
from smilesfeaturizer.model.lgbm_model import train_lgbm
from smilesfeaturizer.constant import (
    ALL_REACTIVE_SITES,
    REACTION_CLASSES_TO_SMART_FRAGMENTS,
    DATAMOL_FEATURES,
)
from smilesfeaturizer.analysis.insight_plotter import (
    calculate_error,
    smiles_insight_plot,
)

__all__ = [
    "train_lgbm",
    "smiles_insight_plot",
    "calculate_error",
    "add_descriptors_to_df",
    "calculate_and_add_ecfp_fingerprints",
    "calculate_and_add_maccs_fingerprints",
    "calculate_and_add_rdkit2d_descriptors",
    "df_scatter_plot",
    "draw_corr",
    "create_inline_dash_dashboard",
    "generate_smiles_feature",
    "sentences2vec",
    "mol2vec_feature",
    "add_molecule_from_smiles",
    "smiles_to_fp",
    "generate_3D_coordinates",
    "smiles_to_image_array",
    "add_all_descriptors",
    "find_reactive_sites",
    "count_reactive_sites",
    "count_reaction_fragments",
    "add_reactive_groups",
    "generate_descriptor_functions",
    "add_chem_properties",
    "expand_reaction_sites",
    "generate_chemical_properties",
    "interpolate_missing_values",
    "extract_extra_features",
    "ALL_REACTIVE_SITES",
    "REACTION_CLASSES_TO_SMART_FRAGMENTS",
    "DATAMOL_FEATURES",
    "perform_pca_on_mol2vec",
    "apply_pca_to_dataframe",
]
__version__ = "0.1.2"
__author__ = "daniel park <parkminwoo1991@gmail.com>"
