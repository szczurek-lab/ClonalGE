import visualization as vis
from clonalGE import clonalGE
import tumoroscope as tum
import pickle
import numpy as np
import random
import os
import pandas as pd
import json
import sys
import distutils.util
import seaborn as sns; sns.set_theme(color_codes=True)
import pre_processing
from scipy.optimize import lsq_linear
from scipy.optimize import minimize
from scipy.stats import nbinom
import warnings

# Suppress specific numpy warnings that are not critical
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=FutureWarning, message='The behavior of DataFrame.sum with axis=None is deprecated')

def main():
    genes_file = 'prostate_data_configs/prostate_top1000_genes.txt'
    plot_colors = ['b','g','r','c','m','y','k','#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    col = ("#efefef",'#003f5c' ,'#AAAAAA' ,'#bc5090' ,'#ff6361', '#ffa600' ,"#47B39C","#74BBFB")

    config = sys.argv[2]
    result_dir = sys.argv[1]
    move_dir = ''

    run_sampling = bool(distutils.util.strtobool(sys.argv[3]))
    save_observed = bool(distutils.util.strtobool(sys.argv[4]))
    vis_observed = bool(distutils.util.strtobool(sys.argv[5]))
    vis_results = bool(distutils.util.strtobool(sys.argv[6]))
    saved_inputs = bool(distutils.util.strtobool(sys.argv[7]))
    progress_bar = bool(distutils.util.strtobool(sys.argv[8]))
    use_negative_binomial = bool(distutils.util.strtobool(sys.argv[9]))
    seed = int(sys.argv[10])   # or random.randint(1, 1000)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        print("Directory ", result_dir, " Created ")


    file_log = open(result_dir + '/log.txt', 'w')
    sys.stdout = file_log

    print('The config file is: ' + config)
    print('seed:', seed)
    with open(config) as json_data_file:
        data = json.load(json_data_file)
    ### making st data structure
    ### can be change by changing the data
    sections_n_file = []
    for name,value in data['observed'].items():
        sections_n_file.append(name)

    def files_to_inputs(data,save_observed):
        st_sections = [[]]*len(sections_n_file)
        counter = 0
        for sections in sections_n_file:
            st_sections[counter] = pd.read_csv(move_dir + data['observed'][sections], delimiter = "\t", dtype={"refContig":"string","refPos":int,"refAllele":"string","base":"string","source":"string", "spot":"string", "cnt":int})
            st_sections[counter]['source'] = sections
            st_sections[counter]['spot'] = sections+'_' + st_sections[counter]['spot']
            counter = counter + 1
        frames = st_sections

        ###########################################################################################################
        ###########################################################################################################


        print("######################  pre processing started: preparing observed variables ####################")

        st_wes = pre_processing.generate_st_wes(frames, move_dir + data['C_variation']['WES_file'],data['criteria']['offset'])
        inputs_dump_dir = move_dir + data['results']['data_configs']
        pickle.dump(st_wes, open(inputs_dump_dir + 'st_wes.pickle', 'wb'))


        #df of selected spots

        n_s_data = pd.read_csv(move_dir + data['n_variation']['n_file'],sep=',')
        n_s_data = n_s_data[~n_s_data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
        n_s_data = n_s_data.drop_duplicates()
        n_s_data = n_s_data[n_s_data['type'].str.contains('Cancer')]
        if(n_s_data.barcode.duplicated().any()==True):
            n_s_data[n_s_data.barcode.duplicated()]=np.nan
            n_s_data = n_s_data[~n_s_data.isin([np.nan, np.inf, -np.inf]).any(axis=1)]
            print("There are duplications in the barcode(probably different types)")
  
        st_wes_selected_spots = st_wes[st_wes['spot'].isin(n_s_data['barcode'])]


        mutations_locations = np.unique(st_wes_selected_spots['gene'])
        mutations_locations_minus1 = np.unique(st_wes_selected_spots['gene_minus1'])
        # the rows are mutation_id and the columns are clones for C_ik
        K, C_ik = pre_processing.generate_C(data['C_variation']['method'], mutations_locations, move_dir + data['C_variation']['phyloWGS'], data['C_variation']['tree'],move_dir + data['C_variation']['selected_canopy_tree'])
        if np.sum(np.sum(C_ik > 1)):
            C_ik[C_ik > 1] = 1
            print("Warning! C has numbers higher than 1")
        F_epsilon,F = pre_processing.generate_F(data['Gamma']['F_epsilon'],move_dir + data['Gamma']['F_file'],K)




        A,D,A_df,D_df = pre_processing.generate_A_D(st_wes_selected_spots,data['criteria']['offset'],n_s_data['barcode'])



        if np.sum(n_s_data['barcode'].isin(A.columns))!=len(n_s_data['barcode']):
            Warning('Out of '+str(len(n_s_data['barcode']))+' spots in the cell number file, '+str(
                np.sum(n_s_data['barcode'].isin(A.columns)))+' number of them showed up in the st file.')
            Warning('Out of ' + str(len(C_ik.index)) + ' mutations in the tree, ' + str(
                np.sum(C_ik.index.isin(A.index))) + ' number of them showed up in the st file.')
            C_ik = C_ik[C_ik.index.isin(A.index)]
            C_ik = C_ik[np.sum(C_ik,axis=1)>0]

            A=A[A.index.isin(C_ik.index)]
            D=D[D.index.isin(C_ik.index)]

            C_ik = C_ik.reindex(A.index)

            n_s_data = n_s_data[n_s_data['barcode'].isin(A.columns)]
            n_s_data = n_s_data.reset_index(drop=True)

        exs = []
        for section in sections_n_file:
            print(f"DEBUG: Processing section {section}")
            expr_data = pd.read_csv(move_dir + data['expression'][section], delimiter="\t", index_col=0)
            print(f"DEBUG: Loaded expression data shape: {expr_data.shape}")
            print(f"DEBUG: Expression data columns (first 5): {expr_data.columns[:5].tolist()}")
            
            in_section = n_s_data[n_s_data['section']==section]
            print(f"DEBUG: Cell count data for {section}: {len(in_section)} spots")
            print(f"DEBUG: Cell count coordinates (first 5): {in_section['coordinates'].head().tolist()}")
            
            # Match barcodes instead of coordinates
            matching_barcodes = expr_data.columns.intersection(in_section['barcode'])
            print(f"DEBUG: Matching barcodes: {len(matching_barcodes)} out of {len(expr_data.columns)}")
            
            if len(matching_barcodes) == 0:
                print(f"ERROR: No matching barcodes found for section {section}!")
                print(f"Expression data columns (first 5): {expr_data.columns[:5].tolist()}")
                print(f"Cell count barcodes (first 5): {in_section['barcode'].head().tolist()}")
                raise ValueError(f"No matching barcodes for section {section}")
            
            expr_data = expr_data[matching_barcodes]
            # No need to rename columns since they already match
            exs.append(expr_data)
        expr_data = pd.concat(exs, axis=1).dropna()
        expr_data = expr_data[expr_data.sum(axis=1)>0]   # excluding genes with expression = 0
        expr_data = expr_data
        n_s_data = n_s_data[n_s_data['barcode'].isin(expr_data.columns)]
        expr_data = expr_data[n_s_data.barcode]   # setting in right order
        with open(genes_file) as f:
            genes = f.readlines()
            genes = [gene.strip() for gene in genes]
        
        print(f"DEBUG: Loaded {len(genes)} genes from {genes_file}")
        print(f"DEBUG: First 5 genes: {genes[:5]}")
        print(f"DEBUG: Expression data shape before filtering: {expr_data.shape}")
        print(f"DEBUG: Expression data index (first 5): {expr_data.index[:5].tolist()}")
        
        # Check which genes are actually in the expression data
        available_genes = [gene for gene in genes if gene in expr_data.index]
        print(f"DEBUG: {len(available_genes)} out of {len(genes)} genes are available in expression data")
        
        if len(available_genes) == 0:
            print("ERROR: No genes from the gene list are found in expression data!")
            print("Available genes in expression data (first 10):", expr_data.index[:10].tolist())
            print("Genes in gene list (first 10):", genes[:10])
            raise ValueError("Gene list mismatch!")
        
        expr_data = expr_data.loc[available_genes]
        sections_array = np.array(expr_data.columns.str[:4])
        Y = np.array(expr_data)
        Y = Y.T

        A = A[n_s_data.barcode]
        D = D[n_s_data.barcode]

        D = D[A.sum(axis=1)>data['criteria']['st_read_limit']]
        A = A[A.sum(axis=1)>data['criteria']['st_read_limit']]
        C_ik = C_ik[C_ik.index.isin(A.index)]


        if save_observed==True:
            lambda_file = open(result_dir+"/n_lambda.txt", "w")
            np.savetxt(lambda_file, n_s_data['nuclei'],fmt='%s')
            a_file = open(result_dir+"/A.txt", "w")
            np.savetxt(a_file, A,fmt='%s')
            d_file = open(result_dir+"/D.txt", "w")
            np.savetxt(d_file, D,fmt='%s')
            s_file = open(result_dir+"/spots_order.txt", "w")
            np.savetxt(s_file, n_s_data['barcode'], fmt='%s')
            m_file = open(result_dir+"/mutations_order.txt", "w")
            C_ik.to_csv(result_dir+'/C_ik.csv', index=False,sep='\t')
            np.savetxt(m_file, np.unique(A_df["i"]), fmt='%s')
            expr_data.to_csv(result_dir+'/expr.csv', index=True,sep='\t')

        n_s_data['nuclei'][n_s_data['nuclei']==0]=1

        A_df = A_df[A_df['s'].isin(A.columns)]
        A_df = A_df[A_df['i'].isin(A.index)]
        D_df = D_df[D_df['s'].isin(D.columns)]
        D_df = D_df[D_df['i'].isin(D.index)]

        pickle.dump(A_df, open(inputs_dump_dir + 'A_df.pickle', 'wb'))
        pickle.dump(D_df, open(inputs_dump_dir + 'D_df.pickle', 'wb'))
        pickle.dump(F, open(inputs_dump_dir + 'F.pickle', 'wb'))
        pickle.dump(F_epsilon, open(inputs_dump_dir + 'F_epsilon.pickle', 'wb'))
        pickle.dump(A, open(inputs_dump_dir + 'A.pickle', 'wb'))
        pickle.dump(D, open(inputs_dump_dir + 'D.pickle', 'wb'))
        pickle.dump(n_s_data, open(inputs_dump_dir + 'n_s_data_final.pickle', 'wb'))
        pickle.dump(C_ik, open(inputs_dump_dir + 'C_ik.pickle', 'wb'))
        pickle.dump(Y, open(inputs_dump_dir + 'Y.pickle', 'wb'))

        textfile = open(result_dir + '/spots_order.txt', "w")
        for element in A.columns:
            textfile.write(element + "\n")
        textfile.close()

        textfile = open(result_dir + '/mutations_order.txt', "w")
        for element in A.index:
            textfile.write(element + "\n")
        textfile.close()

        textfile = open(result_dir + '/genes_order.txt', "w")
        for element in expr_data.index:
            textfile.write(element + "\n")
        textfile.close()

        return F, F_epsilon, A_df, D_df, C_ik, A, D, n_s_data, Y, sections_array


    if saved_inputs==True:
        F, F_epsilon, A_df, D_df, C_ik, A, D, n_s_data, Y = read_saved_files_to_input(data)

    else:
        F, F_epsilon, A_df, D_df, C_ik, A, D, n_s_data, Y, sections_array = files_to_inputs(data,save_observed)
        print("inputs saved!")

    visualization_dir = result_dir+'/visualization_'+data['structure']['section']
    vis_1 = vis.visualization(visualization_dir)
    if vis_observed == True:
        print("###################### Visualization of the observed variables  ####################")
        vis_1.plot_F_gamma(F_epsilon, F, plot_colors)
        vis_1.hist_matrix(A_df['value'].values.flatten(), 50, 'A')
        vis_1.hist_matrix(D_df['value'].values.flatten(), 50, 'D')
        vis_1.gamma(np.array(data['Gamma']['phi_gamma'])[0], np.array(data['Gamma']['phi_gamma'])[1], 'Phi')
        vis_1.heatmap_seaborn(C_ik.to_numpy(), 'C_seaborn', 'clones', 'mutations', False, 0.5)
        g = sns.clustermap(C_ik, cmap="Blues")
        g.ax_row_dendrogram.set_xlim([-0.1, 0.1])  # Small offset to prevent identical xlims
        g.savefig(result_dir + "/C_ik_clustered.png")


    def estimate_parameters(Y, n_lambda, sections_array):
        sections = list(set(sections_array))
        means = [np.mean(Y[sections_array==section]) for section in sections]
        means = np.array(means)
        scaling_factors = np.mean(means) / means
        scaling_factors_array = np.zeros(len(sections_array))
        for section, scaling_factor  in zip(sections, scaling_factors):
            scaling_factors_array[sections_array==section] = scaling_factor
        scaling_factors_array = scaling_factors_array.reshape(-1,1)
        Y = Y * scaling_factors_array
        g = Y.shape[1]
        pred_p = np.mean(Y, axis=0) / np.var(Y, axis=0)
        pred_B = []
        for i in range(g):
            pred_Bg = np.mean(Y[:,i]/n_lambda)
            pred_B.append(pred_Bg)
        pred_beta = np.var(pred_B) / np.mean(pred_B) * 2   # *2 because this method of estimation probably results in too low beta
        pred_alpha = pred_B / pred_beta
        return pred_p, pred_alpha, pred_beta, 1 / scaling_factors_array

    def calc_B(Y, H, N):
        """Calculate B matrix using linear regression (Tumoroscope+LR approach)"""
        K = H.shape[1]
        G = Y.shape[1]
        N = N * np.eye(len(N))
        X = np.matmul(N,H)
        B = np.zeros((K,G))
        for g in range(G):
            if Y[:,g].sum() > 0:
                B[:,g] = lsq_linear(X, Y[:,g], bounds=(0, np.inf)).x
        return B
    
    def calc_B_negative_binomial(Y, H, N):
        """Calculate B matrix using negative binomial regression (Tumoroscope+NB approach)"""
        K = H.shape[1]
        G = Y.shape[1]
        S = Y.shape[0]
        B = np.zeros((K, G))
        
        # Create design matrix: X = N * H (cell counts × clone proportions)
        X = np.zeros((S, K))
        for s in range(S):
            for k in range(K):
                X[s, k] = N[s] * H[s, k]
        
        for g in range(G):
            if Y[:, g].sum() > 0:
                y_g = Y[:, g]
                
                # Negative binomial regression: Y ~ NB(mean, dispersion)
                # where mean = X * B and we need to estimate B and dispersion
                
                def neg_binom_loglik(params):
                    """Negative log-likelihood for negative binomial regression"""
                    B_g = params[:K]  # Clone-specific expression for gene g
                    r = params[K]     # Dispersion parameter (r > 0)
                    
                    # Ensure non-negative expression and positive dispersion
                    if np.any(B_g < 0) or r <= 0:
                        return 1e10
                    
                    # Calculate mean expression for each spot
                    mu = np.dot(X, B_g)
                    
                    # Avoid numerical issues
                    mu = np.maximum(mu, 1e-6)
                    
                    # Negative binomial log-likelihood
                    # P(Y=y) = Γ(y+r)/(Γ(r)Γ(y+1)) * (r/(r+μ))^r * (μ/(r+μ))^y
                    try:
                        # Using scipy's negative binomial PMF
                        loglik = 0
                        for s in range(S):
                            if y_g[s] >= 0:  # Only for non-negative counts
                                loglik += nbinom.logpmf(y_g[s], r, r/(r + mu[s]))
                        return -loglik  # Return negative log-likelihood for minimization
                    except:
                        return 1e10
                
                # Initial parameters: B from linear regression, r=1
                try:
                    # Get initial B from linear regression
                    B_init = lsq_linear(X, y_g, bounds=(0, np.inf)).x
                    r_init = 1.0
                    initial_params = np.concatenate([B_init, [r_init]])
                    
                    # Bounds: B >= 0, r > 0
                    bounds = [(0, None)] * K + [(1e-6, None)]
                    
                    # Optimize
                    result = minimize(neg_binom_loglik, initial_params, bounds=bounds, 
                                    method='L-BFGS-B', options={'maxiter': 1000})
                    
                    if result.success:
                        B[:, g] = result.x[:K]
                    else:
                        # Fallback to linear regression if NB regression fails
                        B[:, g] = lsq_linear(X, y_g, bounds=(0, np.inf)).x
                        
                except Exception as e:
                    # Fallback to linear regression if anything goes wrong
                    B[:, g] = lsq_linear(X, y_g, bounds=(0, np.inf)).x
        
        return B

    result_txt = result_dir + '/' + data['results']['text_result'] + data['structure']['section'] + '.txt'
    result_obj = result_dir + '/' + data['results']['text_result'] + data['structure']['section']

    print("###################### Start of the sampling  ####################")
    if run_sampling is True:
        K = len(C_ik.columns)
        S = len(n_s_data['barcode'])
        g = Y.shape[1]
        print(f'running on {K} clones, {S} spots and {g} genes')
        print(f'Configuration: min_iter={data["sampling"]["min_iter"]}, max_iter={data["sampling"]["max_iter"]}, batch={data["sampling"]["batch"]}')
        print(f'This will take approximately {data["sampling"]["max_iter"]/data["sampling"]["batch"]} batches')
        pred_p, pred_alpha, pred_beta, pred_t = estimate_parameters(Y, n_s_data['nuclei'].astype(float), sections_array)
        print('pred_p:', np.min(pred_p), '-', np.max(pred_p))
        print('pred_alpha:', np.min(pred_alpha), '-', np.max(pred_alpha))
        print('pred_beta:', pred_beta)

        r, q = 0.005, 2   # estimated in previous steps
        tum_0 = tum.tumoroscope(name=result_obj, K=K, S=S, r=r, p=q, I=len(C_ik),
                                avarage_clone_in_spot=data['Z_variation']['avarage_clone_in_spot'], F=F,
                                C=C_ik.to_numpy(), A=A.to_numpy(), D=D.to_numpy(), F_epsilon=F_epsilon,
                                optimal_rate=data['structure']['optimal_rate'], n_lambda=n_s_data['nuclei'].astype(float),
                                gamma=data['theta']['gamma'], pi_2D=data['structure']['pi_2D'],
                                result_txt=result_txt, rp_est_method='my')

        tum_0.gibbs_sampling(seed=seed, min_iter=int(int(data['sampling']['min_iter'])/1.5), 
                             max_iter=int(int(data['sampling']['max_iter'])/2), 
                             burn_in=int(data['sampling']['burn_in']), batch=int(data['sampling']['batch']),
                             simulated_data=None, n_sampling=data['n_variation']['n_sampling'], 
                             F_fraction=data['Gamma']['F_fraction'], theta_variable=data['theta']['theta_variable'],
                             pi_2D=data['structure']['pi_2D'], th=data['Z_variation']['threshold'], 
                             every_n_sample=data['sampling']['every_n_sample'], 
                             changes_batch=data['sampling']['changes_batch'], 
                             var_calculation=False, progress_bar=progress_bar)

        # Choose regression method based on parameter
        if use_negative_binomial:
            print("Using Negative Binomial regression for gene expression prediction...")
            B = calc_B_negative_binomial(Y, tum_0.inferred_H, tum_0.inferred_n)
        else:
            print("Using Linear regression for gene expression prediction...")
            B = calc_B(Y, tum_0.inferred_H, tum_0.inferred_n)
        
        inits = (tum_0.inferred_n, tum_0.inferred_H, tum_0.inferred_G, tum_0.inferred_pi, tum_0.inferred_phi, tum_0.inferred_Z, B)


        cl = clonalGE(name=result_obj, K=K, S=S, g=g, r=r, q=q, I=len(C_ik),
                      avarage_clone_in_spot=data['Z_variation']['avarage_clone_in_spot'], F=F,
                      C=C_ik.to_numpy(), A=A.to_numpy(), D=D.to_numpy(), F_epsilon=F_epsilon,
                      optimal_rate=data['structure']['optimal_rate'], n_lambda=n_s_data['nuclei'].astype(float),
                      pi_2D=data['structure']['pi_2D'],result_txt=result_txt,
                      Y=Y, p_y=pred_p, b_alpha=pred_alpha, b_beta=pred_beta, t=pred_t, inits=inits)

        cl.gibbs_sampling(seed=seed, min_iter=int(data['sampling']['min_iter']),
                          max_iter=int(data['sampling']['max_iter']), batch=int(data['sampling']['batch']),
                          simulated_data=None, n_sampling=data['n_variation']['n_sampling'],
                          F_fraction=data['Gamma']['F_fraction'], pi_2D=data['structure']['pi_2D'],
                          th=data['Z_variation']['threshold'], every_n_sample=data['sampling']['every_n_sample'],
                          changes_batch=data['sampling']['changes_batch'], progress_bar=progress_bar)
        pickle.dump(cl, open(result_obj, 'wb'))
    else:
        print("loading existing object")
        cl = pickle.load(open(result_obj, 'rb'))


    def post_clonalGE(K,cl,spots_order,result_dir,section,visualization_dir,n_s_data,n_lambda,col,vis_1,sections_n_file):

        print("###################### saving and visualizing the results  ####################")
        tree_clones = [f"C{i}" for i in range(1,(K+1))]
        inferred_h = pd.DataFrame(data=cl.inferred_H, index=spots_order, columns=tree_clones).round(3)
        inferred_z = pd.DataFrame(data=cl.inferred_Z, index=spots_order, columns=tree_clones).round(3)
        inferred_p_z = pd.DataFrame(data=cl.inferred_P_Z, index=spots_order, columns=tree_clones).round(3)
        inferred_n = pd.DataFrame(data=cl.inferred_n, index=spots_order, columns=['n']).round(3)

        inferred_h.to_csv(result_dir + '/' + section + '_h.txt', header=tree_clones, sep='\t', mode='w')
        inferred_z.to_csv(result_dir + '/' + section + '_z.txt', header=tree_clones, sep='\t', mode='w')
        inferred_n.to_csv(result_dir + '/' + section + '_n.txt', header=[section], sep='\t', mode='w')

        pre_processing.plot_prior_inferred_n(n_lambda,cl.inferred_n,section,visualization_dir)

        dir_out = 'rerun/Results_numpy_known_params'
        if not os.path.exists(dir_out):
            os.makedirs(dir_out)
        np.save(result_dir + '/' + section + '_h.npy', cl.inferred_H)
        np.save(result_dir + '/' + section + '_b.npy', cl.inferred_B)
        np.save(result_dir + '/' + section + '_n.npy', cl.inferred_n)

        n_s_data_h = pd.merge(left=n_s_data, right=inferred_h, how="left", left_on=['barcode'], right_on=['barcode'])
        n_s_data_z = pd.merge(left=n_s_data, right=inferred_z, how="left", left_on=['barcode'], right_on=['barcode'])

        for n_section in sections_n_file:
            inferred_H_section = n_s_data_h[n_s_data_h['section'] == n_section][tree_clones].reset_index(drop=True)
            inferred_Z_section = n_s_data_z[n_s_data_z['section'] == n_section][tree_clones].reset_index(drop=True)
            n_s_data_temp = n_s_data[n_s_data['section'] == n_section].reset_index(drop=True)
            vis_1.plot_pie_chart_H(n_s_data_temp, inferred_H_section, col, n_section, tree_clones)
            vis_1.plot_pie_chart_HZ(n_s_data_temp, inferred_H_section, inferred_Z_section, col, n_section, tree_clones)
            vis_1.plot_spots_each_clone( n_s_data_temp,inferred_H_section , n_section)

        vis_1.visualizing_inferred_variables(cl)

    if vis_results==True:
        n_s_data['x'] = n_s_data['x'].astype(float)
        n_s_data['y'] = n_s_data['y'].astype(float)
        n_s_data['nuclei'] = n_s_data['nuclei'].astype(float)
        post_clonalGE(len(C_ik.columns),cl,n_s_data['barcode'],result_dir,data['structure']['section'],visualization_dir,n_s_data,n_s_data['nuclei'],col,vis_1,sections_n_file)

    file_log.close()

if __name__=='__main__':
    main()
