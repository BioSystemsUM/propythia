from Bio.SeqIO.FastaIO import SimpleFastaParser
import csv
import pandas as pd

def read_csv(filename):
    arr = []
    with open(filename) as file:
        reader = csv.DictReader(file, delimiter=';')
        for row in reader:
            arr.append(row['sequence'])
    return arr

def read_fasta(filename):
    d = {}
    with open(filename) as handle:
        for key, sequence in SimpleFastaParser(handle):
            sequence = sequence.upper()
            if(sequence != "SEQUENCEUNAVAILABLE"):
                d[key] = sequence
    return d

def write_to_csv(d):
    with open('datasets/ensembl.csv', 'w') as csvfile:
        fieldnames = ['id', 'sequence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in d.items():
            writer.writerow({'id': key, 'sequence': value})

def read_sequences_csv(csv_file):
    d = {}
    with open(csv_file) as file:
        reader = csv.DictReader(file)
        for row in reader:
            d[row['id']] = row['sequence']
    return d

def remove_essential_genes(d):
    essential_genes = read_sequences_csv("datasets/essential_genes_clean_total.csv")
    ensembl_genes = list(set([value for value in d.values()]))
    print("len(ensembl_genes):", len(ensembl_genes))
    print("len(set(ensembl_genes)):", len(set(ensembl_genes)))
    
    # remove from ensembl_genes all genes that are in essential_genes
    counter = 0
    non_essential_dataset = []
    for gene in ensembl_genes:
        if(gene not in essential_genes.values()):
            non_essential_dataset.append(gene)
        
        
        # print progress
        if(counter % 1000 == 0):
            print(counter, "/", len(ensembl_genes))
        counter += 1
    print("len(non_essential_dataset):", len(non_essential_dataset), ", removed:", len(ensembl_genes) - len(non_essential_dataset))

def create_non_essential_sequences(unique_deg_sequences, ensembl_dataset):
    non_essential_sequences = {}
    for key, big_seq in ensembl_dataset.items():
        if(big_seq in unique_deg_sequences):
            non_essential_sequences[key] = big_seq
    return non_essential_sequences

def create_dict_of_occurrences(d):
    result = {}
    for key, value in d.items():
        key = key.split("|")[0]
        if(key in result):
            result[key].append(value)
        else:
            result[key] = [value]
    return result

def main():
    ensembl_dataset = read_fasta("datasets/mart_export.fa")
    print(len(ensembl_dataset.values()), len(set(ensembl_dataset.values()))) 
    
    deg_dataset = pd.read_csv("datasets/essential_genes.csv", sep=';')
    unique_deg_sequences = set(deg_dataset["sequence"])
    print("unique deg sequences:", len(unique_deg_sequences))
    
    eg_seqs = create_non_essential_sequences(unique_deg_sequences, ensembl_dataset)
    print("----------")
    print("eg_seqs", len(eg_seqs.values()))
    print("eg_seqs uniques", len(set(eg_seqs.values())))
    
    first_ids = [i.split("|")[0] for i in eg_seqs.keys()]
    red_flag_ids = set(first_ids)
    
    d = create_dict_of_occurrences(ensembl_dataset)
    
    for i in red_flag_ids:
        del d[i]
            
    print(len(d))
    all_sequences = [j for i in d.values() for j in i]
    print(len(all_sequences))
    print(len(set(all_sequences)))
            
    
    # print(len(ensembl_dataset), len(ensembl_dataset.values()), len(set(ensembl_dataset.values()))) 
    
    
if __name__ == "__main__":
    main()