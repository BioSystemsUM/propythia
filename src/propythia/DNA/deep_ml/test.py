from Bio.SeqIO.FastaIO import SimpleFastaParser
import pandas as pd
import csv


def read_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        return list(reader)


def read_tsv(filename):
    d = {}
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')

        for i in list(reader)[1:]:
            d[i[0]] = i[1]
    return d


def read_fasta(filename):
    d = {}
    with open(filename) as handle:
        for key, sequence in SimpleFastaParser(handle):
            sequence = sequence.upper()
            if(sequence != "SEQUENCEUNAVAILABLE"):
                d[key] = sequence
    print(len(d), "keys", len(d.values()), "seqs", len(set(d.values())), "unique")
    return d


def get_ids(data):
    """
    embl: dict of id4 -> sequence
    hgnc: dict of id4 -> sequence
    """
    embl = {}
    hgnc = {}
    neither = []
    all_seqs = []
    for i in data[1:]:
        if(i[3].startswith("EMBL:")):
            embl[i[3].replace("EMBL:", "")] = i[-1]
        elif(i[3].startswith("HGNC:")):
            hgnc[i[3]] = i[-1]
        else:
            neither.append(i[-1])

        all_seqs.append(i[-1])

    print(len(all_seqs), "seqs,", len(set(all_seqs)), "unique")
    print("EMBL:", len(embl), "ids unique")
    print("HGNC:", len(hgnc), "ids unique")

    return embl, hgnc


def get_ensembl_ids_from_DEG(data):
    embl, hgnc = get_ids(data)
    ligacoes = read_tsv("datasets/ligacoes.tsv")

    ensembl_from_hgnc = []
    for i in hgnc:
        if(i in ligacoes):
            if(ligacoes[i] != ""):
                ensembl_from_hgnc.append(ligacoes[i])
        else:
            print(i, "not in ligacoes")

    print("converted", len(ensembl_from_hgnc), "hgnc ids to ensembl ids")

    ensembl_from_hgnc = set(ensembl_from_hgnc)
    embl = set(embl.keys())

    to_remove = ensembl_from_hgnc.union(embl)
    print("EMBL ids + converted HGNC to EMBL ids:", len(to_remove), "unique")

    return to_remove


def create_dict_of_occurrences(d):
    result = {}
    for key, value in d.items():
        key = key.split("|")[0]
        if(key in result):
            result[key].append(value)
        else:
            result[key] = [value]
    return result


def remove_essential_genes(ensembl_dataset, embl_ids):
    d = create_dict_of_occurrences(ensembl_dataset)
    all_sequences = [j for i in d.values() for j in i]
    print("before removing:", len(d), "keys", len(all_sequences), "seqs", len(set(all_sequences)), "unique")

    for i in embl_ids:
        if(i in d):
            del d[i]

    all_sequences = [j for i in d.values() for j in i]
    print("after removing:", len(d), "keys", len(all_sequences), "seqs", len(set(all_sequences)), "unique")
    return d


def match_sequences_to_DEG(d, deg_data):
    unique_sequences_deg = set([i[-1] for i in deg_data])
    arr = []
    for i in d.values():
        if(i in unique_sequences_deg):
            arr.append(i)
    return arr


def main():
    print("DEG stats")
    print("-" * 50)
    deg_data = read_csv("datasets/essential_genes.csv")
    embl_ids = get_ensembl_ids_from_DEG(deg_data)

    # ----------------------------------------------------------------------
    filename = "datasets/mart_export.fa"
    print()
    print("ENSEMBL stats")
    print("-" * 50)
    ensembl_dataset = read_fasta(filename)
    match_seqs = set(match_sequences_to_DEG(ensembl_dataset, deg_data))

    # Removing essential genes with same seqs

    no_egs = {}
    for key, val in ensembl_dataset.items():
        if(val not in match_seqs):
            no_egs[key] = val
    print("after removing essential genes:", len(no_egs), "keys", len(
        no_egs.values()), "seqs", len(set(no_egs.values())), "unique")

    # Removing essential genes with EMBL + converted HGNC to EMBL ids

    negative_dataset = {}
    for key, val in no_egs.items():
        gene_id = key.split("|")[0]
        if(gene_id not in embl_ids):
            negative_dataset[key] = val

    print("after removing essential genes with EMBL id:", len(negative_dataset), "keys", len(
        negative_dataset.values()), "seqs", len(set(negative_dataset.values())), "unique")

    # Grouping by gene id

    d = create_dict_of_occurrences(negative_dataset)

    all_sequences = [j for i in d.values() for j in i]
    print("after grouping gene stable ids:", len(d), "keys", len(
        all_sequences), "seqs", len(set(all_sequences)), "unique")


if __name__ == "__main__":
    main()
