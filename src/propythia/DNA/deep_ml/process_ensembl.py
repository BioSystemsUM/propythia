from Bio.SeqIO.FastaIO import SimpleFastaParser
import csv

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


def main():
    genes = read_fasta("datasets/mart_export_unspliced.fa")
    print(len(genes.values()), len(set(genes.values()))) 
    write_to_csv(genes)
    
    
if __name__ == "__main__":
    main()