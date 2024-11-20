import warnings

warnings.filterwarnings("ignore")

import scanpy as sc
import torch
import torch.utils.data as data
import numpy as np
import os
import pickle
import pandas as pd
import subprocess
from typing import Optional
from typing import Dict, Tuple
from scanpy import AnnData
import rootutils

rootutils.setup_root(__file__, indicator="pyproject.toml", pythonpath=True)
ROOT_DIR = rootutils.find_root(__file__, indicator="pyproject.toml")
EMBEDDING_DIR = os.path.join(ROOT_DIR, "data", "uce_assets", "protein_embeddings")

MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH = {
    "ESM2": {
        "human": '/vepfs/fs_users/yftc/code/CureALL/datas/lincs/lincs_landmarkgene_esm2_embedding_n978.pt',

        # "human": os.path.join(EMBEDDING_DIR, "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"),
        # "mouse": os.path.join(EMBEDDING_DIR, "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt"),
        # "frog": os.path.join(
        #     EMBEDDING_DIR, "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt"
        # ),
        # "zebrafish": os.path.join(EMBEDDING_DIR, "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt"),
        # "mouse_lemur": os.path.join(EMBEDDING_DIR, "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt"),
        # "pig": os.path.join(EMBEDDING_DIR, "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt"),
        # "macaca_fascicularis": os.path.join(
        #     EMBEDDING_DIR, "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt"
        # ),
        # "macaca_mulatta": os.path.join(EMBEDDING_DIR, "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt"),
    }
}

# extra_species = (
#     pd.read_csv(os.path.join(ROOT_DIR, "data", "uce_assets", "new_species_protein_embeddings.csv"))
#     .set_index("species")
#     .to_dict()["path"]
# )
# MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH["ESM2"].update(extra_species)  # adds new species


def load_gene_embeddings_adata(
    adata: AnnData, species: list, embedding_model: str
) -> Tuple[AnnData, Dict[str, torch.FloatTensor]]:
    """Loads gene embeddings for all the species/genes in the provided data.

    :param data: An AnnData object containing gene expression data for cells.
    :param species: Species corresponding to this adata

    :param embedding_model: The gene embedding model whose embeddings will be loaded.
    :return: A tuple containing:
               - A subset of the data only containing the gene expression for genes with embeddings in all species.
               - A dictionary mapping species name to the corresponding gene embedding matrix (num_genes, embedding_dim).
    """
    # Get species names
    species_names = species
    species_names_set = set(species_names)

    # Get embedding paths for the model
    species_to_gene_embedding_path = MODEL_TO_SPECIES_TO_GENE_EMBEDDING_PATH[embedding_model]
    available_species = set(species_to_gene_embedding_path)

    # Ensure embeddings are available for all species
    if not (species_names_set <= available_species):
        raise ValueError(f"The following species do not have gene embeddings: {species_names_set - available_species}")

    # Load gene embeddings for desired species (and convert gene symbols to lower case)
    species_to_gene_symbol_to_embedding = {
        species: {
            gene_symbol.lower(): gene_embedding
            for gene_symbol, gene_embedding in torch.load(species_to_gene_embedding_path[species]).items()
        }
        for species in species_names
    }

    # Determine which genes to include based on gene expression and embedding availability
    genes_with_embeddings = set.intersection(
        *[set(gene_symbol_to_embedding) for gene_symbol_to_embedding in species_to_gene_symbol_to_embedding.values()]
    )
    genes_to_use = {gene for gene in adata.var_names if gene.lower() in genes_with_embeddings}

    # Subset data to only use genes with embeddings
    adata = adata[:, adata.var_names.isin(genes_to_use)]

    # Set up dictionary mapping species to gene embedding matrix (num_genes, embedding_dim)
    species_to_gene_embeddings = {
        species_name: torch.stack(
            [species_to_gene_symbol_to_embedding[species_name][gene_symbol.lower()] for gene_symbol in adata.var_names]
        )
        for species_name in species_names
    }

    return adata, species_to_gene_embeddings


def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()


class SincleCellDataset(data.Dataset):
    def __init__(
        self,
        expression: torch.tensor,  # Subset to hv genes, count data! cells x genes
        protein_embeddings: torch.tensor,  # same order as expression, also subset genes x pe
        labels: None,  # optional, tensor of labels
        covar_vals: None,  # tensor of covar values or none
    ) -> None:
        super(SincleCellDataset, self).__init__()

        # Set expression
        self.expression = expression

        # row_sums = self.expression.sum(1)  # UMI Counts
        # log_norm_count_adj = torch.log1p(self.expression / (self.expression.sum(1)).unsqueeze(1) * torch.tensor(1000))

        # # Set log norm and count adjusted expression
        # max_vals, max_idx = torch.max(log_norm_count_adj, dim=0)
        # self.expression_mod = log_norm_count_adj / max_vals

        # # Calculate dropout likliehoods of each gene
        # self.dropout_vec = (self.expression == 0).float().mean(0)  # per gene dropout percentages

        # # Set data info
        # self.num_cells = self.expression.shape[0]
        # self.num_genes = self.expression.shape[1]

        # Set optional label info, including categorical covariate index
        self.covar_vals = covar_vals
        self.labels = labels

        # Set protein embeddings
        self.protein_embeddings = protein_embeddings

        self.item_mode = "expression"
        if self.covar_vals is not None:
            self.item_mode = "expression+covar"

    def __getitem__(self, idx):
        if self.item_mode == "expression":
            if isinstance(idx, int):
                if idx < self.num_cells:
                    return self.expression[idx, :]
                else:
                    raise IndexError
            else:
                raise NotImplementedError
        elif self.item_mode == "expression+covar":
            if isinstance(idx, int):
                if idx < self.num_cells:
                    return self.expression[idx, :], self.covar_vals[idx]
                else:
                    raise IndexError
            else:
                raise NotImplementedError

    def __len__(self) -> int:
        return self.num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


def data_to_torch_X(X):
    if isinstance(X, sc.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()


def anndata_to_sc_dataset(
    adata: sc.AnnData,
    species: str = "human",
    labels: list = [],
    covar_col: str = None,
    hv_genes=None,
    embedding_model="ESM2",
):
    # Subset to just genes we have embeddings for
    adata, protein_embeddings = load_gene_embeddings_adata(
        adata=adata, species=[species], embedding_model=embedding_model
    )

    if hv_genes is not None:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hv_genes)  # Expects Count Data

        hv_index = adata.var["highly_variable"]
        adata = adata[:, hv_index]  # Subset to hv genes only

        protein_embeddings = protein_embeddings[species][hv_index]
    else:
        protein_embeddings = protein_embeddings[species]
    expression = data_to_torch_X(adata.X)

    covar_vals = None
    if len(labels) > 0:
        assert (
            covar_col is None or covar_col in labels
        ), "Covar needs to be in labels"  # make sure you keep track of covar column!
        labels = adata.obs.loc[:, labels].values

        if covar_col is not None:
            # we have a categorical label to use as covariate
            covar_vals = torch.tensor(pd.Categorical(adata.obs[covar_col]).codes)
    return (
        SincleCellDataset(
            expression=expression, protein_embeddings=protein_embeddings, labels=labels, covar_vals=covar_vals
        ),
        adata,
    )


def adata_path_to_prot_chrom_starts(adata, dataset_species, spec_pe_genes, gene_to_chrom_pos, offset):
    """
    Given a :path: to an h5ad,
    """
    pe_row_idxs = torch.tensor([spec_pe_genes.index(k.upper()) + offset for k in adata.var_names]).long()
    print(len(np.unique(pe_row_idxs)))

    spec_chrom = gene_to_chrom_pos[gene_to_chrom_pos["species"] == dataset_species].set_index("gene_symbol")

    gene_chrom = spec_chrom.loc[[k.upper() for k in adata.var_names]]

    dataset_chroms = gene_chrom["spec_chrom"].cat.codes  # now this is correctely indexed by species and chromosome
    print("Max Code:", max(dataset_chroms))
    dataset_pos = gene_chrom["start"].values
    return pe_row_idxs, dataset_chroms, dataset_pos


def process_raw_anndata(row, h5_folder_path, npz_folder_path, scp, skip, additional_filter, root):
    path = row.path
    if not os.path.isfile(root + "/" + path):
        print("**********************************")
        print(f"***********{root + '/' + path} File Missing****")
        print("**********************************")
        print(path, root)
        return None

    name = path.replace(".h5ad", "")
    proc_path = path.replace(".h5ad", "_proc.h5ad")
    if skip:
        if os.path.isfile(h5_folder_path + proc_path):
            print(f"{name} already processed. Skipping")
            return None, None, None

    print(f"Proccessing {name}")

    species = row.species
    covar_col = row.covar_col

    ad = sc.read(root + "/" + path)
    labels = []
    if "cell_type" in ad.obs.columns:
        labels.append("cell_type")

    if covar_col is np.nan or np.isnan(covar_col):
        covar_col = None
    else:
        labels.append(covar_col)

    if additional_filter:
        sc.pp.filter_genes(ad, min_cells=10)
        sc.pp.filter_cells(ad, min_genes=25)

    dataset, adata = anndata_to_sc_dataset(ad, species=species, labels=labels, covar_col=covar_col, hv_genes=None)
    adata = adata.copy()

    if additional_filter:
        sc.pp.filter_genes(ad, min_cells=10)
        sc.pp.filter_cells(ad, min_genes=25)

    num_cells = adata.X.shape[0]
    num_genes = adata.X.shape[1]

    adata_path = h5_folder_path + proc_path
    adata.write(adata_path)

    arr = data_to_torch_X(adata.X).numpy()

    print(arr.max())  # this is a nice check to make sure it's counts
    filename = npz_folder_path + f"{name}_counts.npz"
    shape = arr.shape
    print(name, shape)
    fp = np.memmap(filename, dtype="int64", mode="w+", shape=shape)
    fp[:] = arr[:]
    fp.flush()

    if scp != "":
        subprocess.call(["scp", filename, f"{scp}:{filename}"])
        subprocess.call(["scp", adata_path, f"{scp}:{adata_path}"])

    return adata, num_cells, num_genes


def get_species_to_offsets(offset_file: Optional[str] = None):
    import pickle

    if offset_file is None:
        offset_file = os.path.join(ROOT_DIR, "datas", "uce_assets", "species_offsets.pkl")
    else:
        offset_file = offset_file

    return pickle.load(open(offset_file, "rb"))


def get_species_to_pe(embedding_dir: Optional[str] = None):
    """
    Given an embedding directory, return all embeddings as a dictionary coded by species.
    Note: In the current form, this function is written such that the directory needs all of the following species embeddings.
    """
    if embedding_dir is None:
        embedding_dir = EMBEDDING_DIR

    embeddings_paths = {
        # "human": os.path.join(embedding_dir, "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"),
        "human": '/vepfs/fs_users/yftc/code/CureALL/datas/lincs/lincs_landmarkgene_esm2_embedding_n978.pt',
        # "mouse": os.path.join(embedding_dir, "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt"),
        # "frog": os.path.join(
        #     embedding_dir, "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt"
        # ),
        # "zebrafish": os.path.join(embedding_dir, "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt"),
        # "mouse_lemur": os.path.join(embedding_dir, "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt"),
        # "pig": os.path.join(embedding_dir, "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt"),
        # "macaca_fascicularis": os.path.join(
        #     embedding_dir, "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt"
        # ),
        # "macaca_mulatta": os.path.join(embedding_dir, "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt"),
    }
    # extra_species = (
    #     pd.read_csv(os.path.join(ROOT_DIR, "data", "uce_assets", "new_species_protein_embeddings.csv"))
    #     .set_index("species")
    #     .to_dict()["path"]
    # )
    # embeddings_paths.update(extra_species)  # adds new species

    species_to_pe = {species: torch.load(pe_dir) for species, pe_dir in embeddings_paths.items()}

    species_to_pe = {species: {k.upper(): v for k, v in pe.items()} for species, pe in species_to_pe.items()}
    return species_to_pe


def get_spec_chrom_csv(chorm_csv_path: Optional[str] = None):
    """
    Get the species to chrom csv file
    """
    if chorm_csv_path is None:
        chorm_csv_path = os.path.join(ROOT_DIR, "datas", "uce_assets", "species_chrom.csv")
    else:
        chorm_csv_path = chorm_csv_path

    gene_to_chrom_pos = pd.read_csv(chorm_csv_path)
    gene_to_chrom_pos["spec_chrom"] = pd.Categorical(
        gene_to_chrom_pos["species"] + "_" + gene_to_chrom_pos["chromosome"]
    )  # add the spec_chrom list
    return gene_to_chrom_pos


class MultiDatasetSentences(data.Dataset):
    def __init__(
        self,
        sorted_dataset_names,
        shapes_dict,
        dataset_to_protein_embeddings_path: Optional[str] = None,
        datasets_to_chroms_path: Optional[str] = None,
        datasets_to_starts_path: Optional[str] = None,
        npzs_dir: Optional[str] = None,
        pad_length: int = 1536,
        sample_size: int = 1024,
        chrom_token_offset: int = 143574,
    ) -> None:
        super(MultiDatasetSentences, self).__init__()
        # self.xs = {}
        self.num_cells = {}
        self.num_genes = {}
        self.shapes_dict = shapes_dict
        self.total_num_cells = 0
        self.pad_length = pad_length
        self.sample_size = sample_size
        self.chrom_token_offset = chrom_token_offset

        for name in sorted_dataset_names:
            num_cells, num_genes = self.shapes_dict[name]
            # self.xs[name] = X
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets = sorted_dataset_names

        # TODO: preferably not hard-coded here
        self.dataset_to_protein_embeddings = torch.load(dataset_to_protein_embeddings_path)
        with open(datasets_to_chroms_path, "rb") as f:
            self.dataset_to_chroms = pickle.load(f)
        with open(datasets_to_starts_path, "rb") as f:
            self.dataset_to_starts = pickle.load(f)

        self.npzs_dir = npzs_dir

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    cts = np.memmap(
                        os.path.join(self.npzs_dir, f"{dataset}_counts.npz"),
                        dtype="int64",
                        mode="r",
                        shape=self.shapes_dict[dataset],
                    )
                    counts = cts[idx]
                    counts = torch.tensor(counts).unsqueeze(0)
                    weights = torch.log1p(counts)
                    weights = weights / torch.sum(weights)
                    batch_sentences, mask, seq_len, cell_sentences = sample_cell_sentences(
                        counts,
                        weights,
                        dataset,
                        dataset_to_protein_embeddings=self.dataset_to_protein_embeddings,
                        dataset_to_chroms=self.dataset_to_chroms,
                        dataset_to_starts=self.dataset_to_starts,
                        pad_length=self.pad_length,
                        sample_size=self.sample_size,
                        CHROM_TOKEN_OFFSET=self.chrom_token_offset,
                    )
                    return batch_sentences, mask, idx, seq_len, cell_sentences
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class MultiDatasetSentenceCollator(object):
    def __init__(self, pad_length):
        self.pad_length = pad_length

    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))
        mask = torch.zeros((batch_size, self.pad_length))
        cell_sentences = torch.zeros((batch_size, self.pad_length))

        idxs = torch.zeros(batch_size)

        i = 0
        max_len = 0
        for bs, msk, idx, seq_len, cs in batch:
            batch_sentences[i, :] = bs
            cell_sentences[i, :] = cs
            max_len = max(max_len, seq_len)
            mask[i, :] = msk
            idxs[i] = idx

            i += 1

        return batch_sentences[:, :max_len], mask[:, :max_len], idxs, cell_sentences


def sample_cell_sentences(
    counts,
    batch_weights,
    dataset,
    dataset_to_protein_embeddings,
    dataset_to_chroms,
    dataset_to_starts,
    pad_length,
    sample_size,
    CHROM_TOKEN_OFFSET,
    cls_token_idx: int = 3,
    pad_token_idx: int = 0,
    chrom_token_right_idx=2,
):
    dataset_idxs = dataset_to_protein_embeddings[dataset]  # get the dataset specific protein embedding idxs
    cell_sentences = torch.zeros((counts.shape[0], pad_length))  # init the cell representation as 0s
    mask = torch.zeros((counts.shape[0], pad_length))  # start of masking the whole sequence
    chroms = dataset_to_chroms[dataset]  # get the dataset specific chroms for each gene
    starts = dataset_to_starts[dataset]  # get the dataset specific genomic start locations for each gene

    longest_seq_len = 0  # we need to keep track of this so we can subset the batch at the end

    for c, cell in enumerate(counts):
        weights = batch_weights[c].numpy()
        weights = weights / sum(weights)  # RE NORM after mask

        # randomly choose the genes that will make up the sample, weighted by expression, with replacement
        choice_idx = np.random.choice(np.arange(len(weights)), size=sample_size, p=weights, replace=True)
        choosen_chrom = chroms[choice_idx]  # get the sampled genes chromosomes
        # order the genes by chromosome
        chrom_sort = np.argsort(choosen_chrom)
        choice_idx = choice_idx[chrom_sort]

        # sort the genes by start
        new_chrom = chroms[choice_idx]
        choosen_starts = starts[choice_idx]

        ordered_choice_idx = np.full((pad_length), cls_token_idx)  # start with cls
        # i= 0 first token is CLS
        i = 1  # continue on to the rest of the sequence with left bracket being assumed.
        # Shuffle the chroms now, there's no natural order to chromosomes
        uq_chroms = np.unique(new_chrom)
        np.random.shuffle(uq_chroms)  # shuffle

        # This loop is actually just over one cell
        for chrom in uq_chroms:
            # Open Chrom token
            ordered_choice_idx[i] = (
                int(chrom) + CHROM_TOKEN_OFFSET
            )  # token of this chromosome # i = 1 next token is a chrom open
            i += 1
            # now sort the genes by start order within the chroms
            loc = np.where(new_chrom == chrom)[0]
            sort_by_start = np.argsort(choosen_starts[loc])  # start locations for this chromsome

            to_add = choice_idx[loc[sort_by_start]]
            ordered_choice_idx[i : (i + len(to_add))] = dataset_idxs[to_add]
            i += len(to_add)
            ordered_choice_idx[i] = chrom_token_right_idx  # add the chrom sep again
            i += 1  # add the closing token again

        longest_seq_len = max(longest_seq_len, i)
        remainder_len = pad_length - i

        cell_mask = torch.concat(
            (
                torch.ones(i),
                # pay attention to all of these tokens, ignore the rest!
                torch.zeros(remainder_len),
            )
        )

        mask[c, :] = cell_mask

        ordered_choice_idx[i:] = pad_token_idx  # the remainder of the sequence
        cell_sentences[c, :] = torch.from_numpy(ordered_choice_idx)

    cell_sentences_pe = cell_sentences.long()  # token indices

    return cell_sentences_pe, mask, longest_seq_len, cell_sentences
