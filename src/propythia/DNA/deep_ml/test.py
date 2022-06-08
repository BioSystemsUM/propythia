from Bio.SeqIO.FastaIO import SimpleFastaParser
import pandas as pd

def read_fasta(filename):
    d = {}
    with open(filename) as handle:
        for key, sequence in SimpleFastaParser(handle):
            sequence = sequence.upper()
            if(sequence != "SEQUENCEUNAVAILABLE"):
                d[key] = sequence
    return d

def create_dict_of_occurrences(d):
    result = {}
    for key, value in d.items():
        key = key.split("|")[0]
        if(key in result):
            result[key].append(value)
        else:
            result[key] = [value]
    return result

deg_dataset = pd.read_csv("datasets/essential_genes.csv", sep=';')
print(deg_dataset.shape)

# filter rows that dont have "EMBL:" in id4
deg_dataset = deg_dataset[deg_dataset["id4"].str.contains("EMBL:")]

print(deg_dataset.shape)

# remove repeated sequences
unique_ensembl_ids = [i.replace("EMBL:","") for i in deg_dataset["id4"].unique()]

print(len(unique_ensembl_ids))

# ------------------------------------------------------------------------------------
# unspliced

# ensembl_dataset = pd.read_csv("datasets/ensembl.csv", sep=',')
# print(ensembl_dataset.shape)

# ensembl_dataset["id"] = [i.split("|")[0] for i in ensembl_dataset["id"]]

# # filter rows which id is in unique_ensembl_ids
# ensembl_dataset = ensembl_dataset[~ensembl_dataset["id"].isin(unique_ensembl_ids)]

# print(ensembl_dataset.shape) # 4325

# print(ensembl_dataset.head())

# ------------------------------------------------------------------------------------
# not unspliced
ensembl_dataset = read_fasta("datasets/mart_export.fa")
print(len(ensembl_dataset.values()), len(set(ensembl_dataset.values()))) 

d = create_dict_of_occurrences(ensembl_dataset)
print("len(d):", len(d))

for i in unique_ensembl_ids:
    if(i in d):
        del d[i]
    
print("len(d):", len(d))
all_sequences = [j for i in d.values() for j in i]
print(len(all_sequences), len(set(all_sequences)))