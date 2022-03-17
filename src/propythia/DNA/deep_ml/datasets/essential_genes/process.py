import glob
from Bio.SeqIO.FastaIO import SimpleFastaParser
import csv


def read_fasta(filename):
    d = {}
    with open(filename) as handle:
        for values in SimpleFastaParser(handle):
            d[values[0]] = values[1]
    return d


def read_all_fasta():
    d = {}
    for filename in glob.glob("fasta/*"):
        d[filename[6:-6]] = read_fasta(filename)
    return d


def read_csv(filename):
    file = open(filename)
    csvreader = csv.reader(file, delimiter=';')
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    return rows


def create_dataset(outfile, sequences, annotations):
    mat = []
    for i in annotations:
        new_line = i + [sequences[i[0]][i[1]]]
        mat.append(new_line)

    with open(outfile, 'w') as f:
        csv.writer(f, delimiter=';').writerows(mat)


if __name__ == '__main__':
    sequences = read_all_fasta()
    annotations = read_csv("deg_annotation_e.csv")
    create_dataset("dataset.csv", sequences, annotations)
