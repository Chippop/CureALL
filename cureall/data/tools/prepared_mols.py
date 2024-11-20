import os
import pickle
from multiprocessing import Pool

import lmdb
import numpy as np
import pandarallel
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

# 禁用RDKit日志
RDLogger.DisableLog("rdApp.*")

n_cpus = os.cpu_count()
pandarallel.pandarallel.initialize(progress_bar=True, nb_workers=int(72))


def smi2_2Dcoords(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), f"2D coordinates shape is not align with {smi}"
    return coordinates


def smi2_3Dcoords(smi, cnt):
    mol = Chem.MolFromSmiles(smi)
    mol = AllChem.AddHs(mol)
    coordinate_list = []
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(
                mol, randomSeed=seed
            )  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)  # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    # print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)

            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(smi)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)  # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    # print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(smi)
        except:
            # print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(smi)

        assert len(mol.GetAtoms()) == len(coordinates), f"3D coordinates shape is not align with {smi}"
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def inner_smi2coords(content):
    smi = content
    cnt = 10  # conformer num,all==11, 10 3d + 1 2d

    mol = Chem.MolFromSmiles(smi)
    if len(mol.GetAtoms()) > 400:
        coordinate_list = [smi2_2Dcoords(smi)] * (cnt + 1)
        # print("atom num >400,use 2D coords", smi)
    else:
        coordinate_list = smi2_3Dcoords(smi, cnt)
        # add 2d conf
        coordinate_list.append(smi2_2Dcoords(smi).astype(np.float32))
    mol = AllChem.AddHs(mol)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H
    return pickle.dumps({"atoms": atoms, "coordinates": coordinate_list, "smi": smi}, protocol=-1)


def smi2coords(content):
    if content is None:
        return None
    try:
        return inner_smi2coords(content)
    except:
        return None


def prepare_smiles(df):
    df["mols_dict"] = df["smi"].parallel_apply(smi2coords)
    return df


def save_to_lmdb(df, outpath):
    if os.path.exists(outpath):
        print(f"{outpath} already exists, remove it? (y/n)")
        if input().lower() == "y":
            os.remove(outpath)
        else:
            return
    env_new = lmdb.open(
        outpath,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    i = 0
    for _, record in tqdm(enumerate(df["mols_dict"])):
        if record is not None:
            txn_write.put(f"{i}".encode("ascii"), record)
            i += 1
    print(f"process {i} lines")
    txn_write.commit()
    env_new.close()


def write_lmdb(smiles_list, seed=42, outpath="./results", nthreads=8):
    if os.path.exists(outpath):
        print(f"{outpath} already exists, remove it? (y/n)")
        if input().lower() == "y":
            os.remove(outpath)
        else:
            return
    env_new = lmdb.open(
        outpath,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )
    txn_write = env_new.begin(write=True)
    with Pool(nthreads) as pool:
        i = 0
        for inner_output in tqdm(pool.imap(smi2coords, smiles_list)):
            if inner_output is not None:
                txn_write.put(f"{i}".encode("ascii"), inner_output)
                i += 1
        print(f"process {i} lines")
        txn_write.commit()
        env_new.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    args = parser.parse_args()

    # df = pd.read_csv(args.input)
    # df = prepare_smiles(df)
    # save_to_lmdb(df, args.output)
    # smiles_list = df["canonical_smiles"].tolist()
    import json

    smiles_list = json.load(open(args.input))
    valid_smiles = []
    print(len(smiles_list))

    for smi in tqdm(smiles_list, desc="validating smiles", total=len(smiles_list)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"invalid smi: {smi}")
            else:
                valid_smiles.append(smi)
        except:
            print(f"invalid smi: {smi}")
            continue

    print(f"valid smiles: {len(valid_smiles)}")
    write_lmdb(valid_smiles, seed=42, outpath=args.output, nthreads=72)
