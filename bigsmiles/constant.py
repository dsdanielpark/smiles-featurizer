
REACTION_CLASSES_TO_SMILES_FRAGMENTS = {
                                        'Cytochrome P450 - Dependent Oxidations': ['C=O', 'CC'],
                                        'Dechlorination': ['CCl', 'CX'],  
                                        'Amine Oxidations': ['CN', 'CCN'],
                                        'Dehydrogenations': ['CC', 'C=C'],
                                        'Azo Reductions': ['N=N'],  
                                        'Nitro Reductions': ['[N+](=O)[O-]'],  
                                        'Carbonyl Reductions': ['C=O'],  
                                        'Ester Hydrolysis': ['C(=O)O'],  
                                        'Amides Hydrolysis': ['C(=O)N'], 
                                        'Oxidation Cytochrome P450 - Dependent Oxidations: Aromatic Hydroxylations': ['c'],  
                                        'Aliphatic Hydroxylations': ['CC'],
                                        'Epoxidation': ['C=C'],  
                                        'Oxidative Dealkylation, N-Dealkylation': ['CN', 'CCN'],
                                        'Oxidative Dealkylation, O-Dealkylation': ['CO', 'CCO'],
                                        'Oxidative Dealkylation, S-Dealkylation': ['CS', 'CCS'],
                                        'N-Oxidation, primary amines': ['CN'],
                                        'N-Oxidation, secondary amines': ['CCN'],
                                        'N-Oxidation, tertiary amines': ['CCCN'],
                                        'S-Oxidation': ['CS'],
                                        'Deamination': ['CN'],
                                        'Desulfuration': ['CS'],
                                    }

REACTION_CLASSES_TO_SMART_FRAGMENTS = {
                                        'Amine_Oxidation': ['[NH2]', '[NHR]', '[NR2]'],
                                        'Ester_Hydrolysis': ['[#6][C](=[O])[O][#6]'],
                                        'Carboxylation': ['C(=O)O'],
                                        'Dehalogenation': ['[F,Cl,Br,I]'],
                                        'Alcohol_Oxidation': ['[OH]'],
                                        'Epoxidation': ['C=C'],
                                        'Glucuronidation': ['O', 'c1ccc(O)cc1', 'C(=O)O'],
                                        'Phosphorylation': ['P(=O)(O)(O)O'],
                                        'Sulfation': ['O', 'N', 'c1ccc(O)cc1'],
                                        'Methylation': ['N', '[SH]', 'O']
                                    }

ALL_REACTIVE_SITES = {
                        'Nitro_Reduction': '[N+](=O)[O-]',
                        'Carbonyl_Reduction': 'C=O',
                        'Dehydration': '[OH]',
                        'Amides': 'C(=O)N',
                        'Oxidations': '[OH]',
                        'Epoxidation': 'CC=C',
                        'Oxidative_Dealkylation': 'COC',
                        'Cytochrome P450 - Dependent Oxidations': 'C=O',
                        'Dechlorination': 'CCl',
                        'Amine Oxidations': 'CN',
                        'Dehydrogenations': 'CC',
                        'Azo Reductions': 'N=N',
                    }


DATAMOL_FEATURES = [
    'maccs', 'avalon', 'ecfp', 'fcfp', 'topological', 'atompair', 
    'rdkit', 'pattern', 'layered', 'secfp', 'erg', 'estate', 
    'avalon-count', 'rdkit-count', 'ecfp-count', 'fcfp-count', 
    'topological-count', 'atompair-count', 'cats2D', 'pharm2D', 
    'scaffoldkeys', 'skeys'
]
