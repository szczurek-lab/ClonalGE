import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy import sparse


################## Prostate data
data = pd.read_csv('files_variables/prostate_H_and_N.csv', index_col=0)
expr_data = pd.read_csv('files_variables/prostate_Y_raw.csv', index_col=0)

mutation=True
name=''

if mutation==True:
    mut_p = pd.read_csv('prostate_genes.txt')
    mut_p.columns=['gene']
    expr_data.index = pd.DataFrame(expr_data.index,columns=['gene'])['gene'].str.split(' ', expand=True)[0]
    expr_data = expr_data[expr_data.index.isin(mut_p['gene']).tolist()]
    name='mut'


Y = np.array(expr_data)
Y = Y.T

H = np.array(data[['H_C1', 'H_C2', 'H_C3', 'H_C4']])
N = np.array(data['all'])

print(N.shape, H.shape, Y.shape)
S, G = Y.shape
K = H.shape[1]
print(S, G, K)

N = N * np.eye(len(N))
X = np.matmul(N,H)



th_spot = 5000
# raw (no scaling)
B = np.zeros((K,G))
for g in range(G):
    if sum(Y[:,g]) > 0 and np.sum(Y[:,g]>0)>th_spot:
        B[:,g] = lsq_linear(X[Y[:,g]>0,:], Y[Y[:,g]>0,g], bounds=(0, np.inf)).x


# saving B for GSEA
B_df = pd.DataFrame(columns=['NAME','DESCRIPTION',1,2,3,4])
B_df['NAME'] = expr_data.index.str.split().str[0]
B_df['DESCRIPTION'] = 'na'
for col in range(1,5):
    B_df[col] = B[col-1]
B_df.to_csv('Bp_raw_'+str(th_spot)+name+'.txt', index=False, sep='\t')


# scaled
sections = ['P1.2', 'P2.4', 'P3.3']
means = [np.mean(np.mean(expr_data.filter(regex=section))) for section in sections]
means = np.array(means)
scaling_factors = np.mean(means) / means
print(scaling_factors)

Y_scaled = np.array(expr_data)
Y_scaled = Y_scaled.T
for i, section in enumerate(sections):
    in_section = np.where(expr_data.columns.str.startswith(section))[0]
    Y_scaled[in_section] = Y_scaled[in_section] * scaling_factors[i]

B_scaled = np.zeros((K,G))
for g in range(G):
    if sum(Y_scaled[:,g]) > 0 and np.sum(Y[:,g]>0)>th_spot:
        B_scaled[:,g] = lsq_linear(X[Y_scaled[:,g]>0,:], Y_scaled[Y_scaled[:,g]>0,g], bounds=(0, np.inf)).x

# saving B for GSEA
B_df = pd.DataFrame(columns=['NAME','DESCRIPTION',1,2,3,4])
B_df['NAME'] = expr_data.index.str.split().str[0]
B_df['DESCRIPTION'] = 'na'
for col in range(1,5):
    B_df[col] = B_scaled[col-1]
B_df.to_csv('Bp_scaled_'+str(th_spot)+name+'.txt', index=False, sep='\t')


################## Breast data
data = pd.read_csv('files_variables/breast_H_and_N.csv', index_col=0)

with open('files_variables/breast_spots.txt') as f:
	spots = f.readlines()
spots = [spot.strip() for spot in spots]

with open('files_variables/breast_genes.txt') as f:
	genes = f.readlines()
genes = [gene.strip() for gene in genes]


sparse_Y = sparse.load_npz('files_variables/breast_Y_raw.npz')
Y = sparse_Y.toarray()

if mutation==True:
    Y_df = pd.DataFrame(Y,columns=genes)
    mut = pd.read_csv('breast_genes.txt')
    mut.columns=['gene']
    Y_df = Y_df[Y_df.columns[Y_df.columns.isin(mut['gene'])]]

    genes = Y_df.columns
    Y = Y_df.to_numpy()
    name='mut'


H = np.array(data.filter(regex='^H_'))
N = np.array(data['all'])

S, G = Y.shape
K = H.shape[1]
print(S, G, K)

N = N * np.eye(len(N))
X = np.matmul(N,H)

# raw (no scaling)
B = np.zeros((K,G))
for g in range(G):
    if sum(Y[:,g]) > 0 and np.sum(Y[:,g]>0)>th_spot:
        B[:,g] = lsq_linear(X[Y[:,g]>0,:], Y[Y[:,g]>0,g], bounds=(0, np.inf)).x

# saving B for GSEA
B_df = pd.DataFrame(columns=['NAME','DESCRIPTION',1,2,3,4,5,6,7])
B_df['NAME'] = genes
B_df['DESCRIPTION'] = 'na'
for col in range(1,8):
    B_df[col] = B[col-1]
B_df.to_csv('Bb_raw_'+str(th_spot)+name+'.txt', index=False, sep='\t')


# scaled
sections = ['112_C1', '112_D1', 
            '113_A1', '113_B1', 
            '114_C1', '114_D1']

means = [np.mean(Y[np.char.startswith(spots, section)]) for section in sections]
means = np.array(means)
scaling_factors = np.mean(means) / means

Y_scaled = Y
for i, section in enumerate(sections):
    in_section = np.char.startswith(spots, section)
    Y_scaled[in_section] = Y_scaled[in_section] * scaling_factors[i]
    
B_scaled = np.zeros((K,G))
for g in range(G):
    if sum(Y_scaled[:,g]) > 0 and np.sum(Y[:,g]>0)>th_spot:
        B_scaled[:,g] = lsq_linear(X[Y_scaled[:,g]>0,:], Y_scaled[Y_scaled[:,g]>0,g], bounds=(0, np.inf)).x

# saving B for GSEA
B_df = pd.DataFrame(columns=['NAME','DESCRIPTION',1,2,3,4,5,6,7])
B_df['NAME'] = genes
B_df['DESCRIPTION'] = 'na'
for col in range(1,8):
    B_df[col] = B_scaled[col-1]
B_df.to_csv('Bb_scaled_'+str(th_spot)+name+'.txt', index=False, sep='\t')

