# earlier:
# pip install pyensembl
# pyensembl install --release 84 --species human
from pyensembl import EnsemblRelease
import pandas as pd

def get_genes_at_locus(chromosome, position, release=84):
    data = EnsemblRelease(release)
    return data.gene_names_at_locus(contig=chromosome, position=position)

# genes related to prostate cancer
# now: genes_file is JSON file downloaded from https://portal.gdc.cancer.gov
def rel_to_cancer(gene_name, genes_file):
    with open(genes_file) as f:
        content = f.read()
        if '\"' + gene_name + '\"' in content: 
            return True
        return False

# print(get_genes_at_locus(20, 50940000))

def read_vcf(file_name):
    with open(file_name) as f:
        for line in f:
            if line[0]=='#' and line[1]!='#':
                col_names = line[1:].split()
                break
    return pd.read_csv(file_name, sep='\t', comment='#', header=None, names = col_names, dtype={'CHROM': str})


def read_ac(file_name):
    data = pd.read_csv(file_name, sep='\t')
    data['refPos']+=1  # to compare with WES positions; it prints log 'INFO:numexpr.utils:NumExpr defaulting to 4 threads'
    return data

def filter_mutations(data_wes, data_st):  
# returns a set of tuples (chromosome, position, ref_base, alt_base) for mutations that are both in st and wes (not in germline)
    #data_wes = data_wes.loc[~data_wes['INFO'].str.contains("Germline")]  # removing germline mutations
    data_wes = data_wes.loc[data_wes['INFO'].str.contains("Somatic")]  # leaving only 'LikelySomatic' and 'StrongSomatic'
    wes_mutations = set([tuple(row) for row in data_wes[['CHROM','POS','REF','ALT']].itertuples(index=False)])
    st_mutations = set([tuple(row) for row in data_st[['refContig','refPos','refAllele','base']].itertuples(index=False)])
    return wes_mutations.intersection(st_mutations)


def filter_df(data, data_type, mutations):  # limit wes/st data to rows with specific mutations
    d = data.copy()
    d['save'] = [0]*len(d)
    for index, row in d.iterrows():
        if data_type=='wes':
            if tuple(row[['CHROM','POS','REF','ALT']]) in mutations:
                d.loc[index, 'save'] = 1
        elif data_type=='st':
            if tuple(row[['refContig','refPos','refAllele','base']]) in mutations:
                d.loc[index, 'save'] = 1
    return data.loc[d['save']==1]

def write_st(data, name):
	data.to_csv(name, sep='\t', index=False)

# example:
# data_wes = read_vcf('wes2.4_bwa_vardict_RAW_SNP_filter.AF0.02_testsomatic_2vcfsomatic.vcf')
# data_st = read_ac('output_vardic_stephani.ac')
# mutations = filter_mutations(data_wes, data_st)
# cut_st = filter_df(data_st, 'st', mutations)
# write_st(cut_st, 'ST_cut.ac')
