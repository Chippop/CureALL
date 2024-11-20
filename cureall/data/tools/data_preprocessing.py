import os
from typing import Union

import numpy as np
import pandas as pd

from .prepared_mols import smi2coords


def concatenated_embedding(lincs_filtered, drug_embedding, cell_embedding):
    # vector mapping version
    drug_embeddings = drug_embedding.loc[lincs_filtered["cmap_name"]].values
    cell_embeddings = cell_embedding.loc[lincs_filtered["cell_iname"]].values

    concatenated_embeddings = np.concatenate([drug_embeddings, cell_embeddings], axis=1)

    drug_cell_embeddings = pd.DataFrame(concatenated_embeddings)

    return drug_cell_embeddings.values


def parser_LINCS_dataset(data_root: str) -> Union[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"{data_root} does not exist.")

    lincs_result_path = os.path.join(data_root, "lincs2020", "lincs2020_treat_filtered.txt")
    lincs_result = pd.read_csv(lincs_result_path, sep="\t")
    index_file_path = os.path.join(data_root, "lincs2020", "filtered_indices.csv")
    filtered_indices = pd.read_csv(index_file_path, header=None).squeeze()
    filtered_indices = filtered_indices.astype(int)
    lincs_result = lincs_result.drop(index=filtered_indices, errors="ignore")
    lincs_result = lincs_result.reset_index(drop=True)
    lincs_control_path = os.path.join(data_root, "lincs2020", "lincs2020_control.txt")
    lincs_control = pd.read_csv(lincs_control_path, sep="\t")

    lincs_merged = lincs_result.merge(lincs_control, on="cell_iname", suffixes=("_with_drug", "_without_drug"))

    lincs_result_value = lincs_merged.iloc[:, 2:].values

    cell_embedding_path = os.path.join(data_root, "cell", "lincs_counts_embedding_UCE.csv")
    cell_embedding = pd.read_csv(cell_embedding_path, index_col=0)
    lincs_cell_embedding = cell_embedding

    lincs_drug_embedding_path = os.path.join(data_root, "drug", "lincs2_drug_embeddings_unimol.csv")
    lincs_drug_embedding = pd.read_csv(lincs_drug_embedding_path, index_col=0)
    lincs_cell_drug_embedding = concatenated_embedding(lincs_result, lincs_drug_embedding, lincs_cell_embedding)

    lincs_train_input_data = lincs_cell_drug_embedding
    lincs_train_output_data = lincs_result_value

    return lincs_train_input_data, lincs_train_output_data


def parser_LINCS_dataset_raw(data_root: str) -> Union[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"{data_root} does not exist.")

    lincs_result_path = os.path.join(data_root, "lincs2020", "lincs2020_treat_filtered.txt")
    lincs_result = pd.read_csv(lincs_result_path, sep="\t")
    index_file_path = os.path.join(data_root, "lincs2020", "filtered_indices.csv")
    filtered_indices = pd.read_csv(index_file_path, header=None).squeeze()
    filtered_indices = filtered_indices.astype(int)
    lincs_result = lincs_result.drop(index=filtered_indices, errors="ignore")
    lincs_result = lincs_result.reset_index(drop=True)
    lincs_control_path = os.path.join(data_root, "lincs2020", "lincs2020_control.txt")
    lincs_control = pd.read_csv(lincs_control_path, sep="\t")
    lincs_label = lincs_result.merge(lincs_control, on="cell_iname", suffixes=("_with_drug", "_without_drug"))
    lincs_label_value = lincs_label.iloc[:, 2:].values

    # get cell_name to uce_embedding mapping
    cell_embedding_path = os.path.join(data_root, "cell", "lincs_counts_embedding_UCE.csv")
    cell_embedding = pd.read_csv(cell_embedding_path)
    values_array = cell_embedding.iloc[:, 1:].to_numpy()
    cell_iname_list = cell_embedding["cell_iname"].tolist()
    cell_iname_to_uce_embedding = dict(zip(cell_iname_list, values_array))

    lincs_drug_smiles_path = os.path.join(data_root, "drug", "lincs2_smile.csv")
    lincs_drug_smiles = pd.read_csv(lincs_drug_smiles_path, index_col=0)
    lincs_drug_smiles["mol_dict"] = lincs_drug_smiles["canonical_smiles"].parallel_apply(smi2coords)
    lincs_drug_smiles.to_csv(os.path.join(data_root, "drug", "lincs2_smile_with_mol_dict.csv"))
    # create mapping from cmap_name to cononical_smile
    # lincs_drug_mapping = lincs_drug_smiles["canonical_smiles"].to_dict()
    # lincs_label["smi"] = lincs_label["cmap_name"].map(lincs_drug_mapping)
    # boundary = lincs_label_value.shape[-1] // 2
    # lincs_control_value = lincs_label_value[:, boundary:]
    # lincs_output_value = lincs_label_value[:, :boundary]

    # lincs_data = []
    # for i in range(len(lincs_label)):
    #     cmap_name = lincs_label["cmap_name"][i]
    #     smi = lincs_drug_mapping[cmap_name]
    #     lincs_data.append({
    #         "smiles": lincs_label["cmap_name"][i],

    #     })
    return lincs_label
