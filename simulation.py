import numpy as np

class simulation:
    def __init__(self,K,S,g,r,q,I,F,D,A,C,avarage_clone_in_spot,random_seed,F_epsilon,n,p_c_binom,theta,Z,n_lambda,
                 F_fraction,pi_2D,Y,p_y,b_alpha,b_beta,b_alpha_shape=None,b_alpha_scale=None):
        self.K=K
        self.S=S
        self.g=g
        self.r=r
        self.q=q
        self.I=I
        self.avarage_clone_in_spot=avarage_clone_in_spot#2
        self.alpha=(avarage_clone_in_spot*K)/(K-avarage_clone_in_spot)
        self.F_epsilon = F_epsilon
        self.beta = F_epsilon[0][1]
        self.p_c_binom = p_c_binom
        self.theta = theta
        np.random.seed(random_seed)
        # we can define n for the simulated data or let it be generated usinig it's distribution
        if n is None:
            self.generate_n(n_lambda)
        else:
            self.n = n

        if p_y is None:
            self.p_y = np.random.random(g)
        else:
            self.p_y = p_y
        if b_alpha_shape is None:
            self.b_alpha_shape = 1
        else:
            self.b_alpha_shape = b_alpha_shape
        if b_alpha_scale is None:
            self.b_alpha_scale = 0.2
        else:
            self.b_alpha_scale = b_alpha_scale
        if b_alpha is None:
            self.b_alpha = np.random.gamma(b_alpha_shape, b_alpha_scale, g)
        else:
            self.b_alpha = b_alpha
        self.b_beta = b_beta

        self.calculating_alpha_s(n_lambda, avarage_clone_in_spot, K)

        # we can give F as input for the simulated data or let it be random
        if F_fraction is True:
            self.generate_F(self.K,self.beta)
        else:
            self.F = np.array(F)

        # we can define C for the simulated data or let it be generated using its distribution
        if C is None:
            self.generate_C(self.I, self.K,self.p_c_binom)
        else:
            self.C = C

        if pi_2D is True:
            self.generate_pi_2D(self.zeta_s, self.K)
        else:
            self.generate_pi(self.alpha, self.K)

        self.generate_phi(self.r,self.q,self.I,self.K)

        if Z is None:
            self.generate_Z(self.K, self.pi,pi_2D)
        else:
            self.Z = Z

        self.generate_G(self.F, self.S, self.K, self.F_epsilon, self.Z)
        self.generate_H(self.G, self.S, self.K, self.Z)


        # We can give D as input or we can generate it by it's distribution
        if D is None:
            self.generate_D(I=self.I, S=self.S, K=self.K, phi=self.phi, H=self.H, n=self.n)
        else:
            self.D = D

        if A is None:
            self.generate_A(self.I, self.S, self.H, self.C, self.D,self.phi,self.theta)
        else:
            self.A = A

        self.generate_B(self.b_alpha, self.b_beta, self.K, self.g)
        if Y is None:
            self.generate_Y(self.n, self.H, self.B, self.p_y, self.S, self.g)
        else:
            self.Y = Y
        print("simulation finished")


    def calculating_alpha_s(self,n_lambda,avarage_clone_in_spot,K):
        density = (n_lambda) / (np.max(n_lambda))
        self.avarage_clone_in_spot_s = np.maximum(1,np.minimum(n_lambda,density * avarage_clone_in_spot))
        self.zeta_s = (self.avarage_clone_in_spot_s*K)/(K-self.avarage_clone_in_spot_s)


    def generate_n(self,n_lambda):
        self.n = np.random.poisson(n_lambda)
        self.n[self.n==0] = (n_lambda[self.n==0])
        #while np.sum(self.n==0)>0:
        #    self.n[self.n==0] = np.random.poisson(n_lambda[self.n==0])

    # Here we use F for multiplying the fractions to a constant value
    # For example if we do not want to have F as input, then F_fraction would be true
    # and here we generate some fractions using Dirichlet distribution but gamma( 0.2,...)
    # would not generate big numbers and because we want the numbers to be very larger than
    # F_epsilon, then we need a constant like 40 (F[0]) to got multiplied by the 0.2
    # also for the second parameter of the gamma distribution, we need a number which is given
    # by F[0][1]
    # TODO: I need to re-code this part with introducing new variable, this is ugly
    def generate_F(self,K,beta):
        temp = np.random.dirichlet(np.ones(K))
        self.F = []
        for k in range(K):
            self.F.append([temp[k], beta])
        self.F = np.array(self.F)


    # I am here generating c_i in all k with arbitrary probabilities of each k would be zero.
    # based on the experiance I just put the more probability for K's of the leaf nodes
    def generate_C(self,I,K,p_c_binom):
        #self.C = np.random.binomial(size=(I,K), n=1, p=np.random.beta(0.5, 1, size=K))
        self.C = np.zeros((I,K),dtype=np.float64)
        for i in range(I):
            half = (np.random.binomial(size=(K-1), n=1, p=p_c_binom)+1)/ 2
            self.C[i][:(K-1)] = np.random.binomial(size=(K-1), n=1, p=p_c_binom) *half
            while self.C[i][:(K-1)].sum()==0:
                half = (np.random.binomial(size=(K-1), n=1, p=p_c_binom)+1)/ 2
                self.C[i][:(K-1)] = np.random.binomial(size=(K-1), n=1, p=p_c_binom) *half
        if(sum(self.C.sum(axis=1)==0)>0):
            print(self.C.sum(axis=1)==0)
            raise Exception("A row in C is zero(0)")
        #print(C)

    def generate_pi(self,zeta,K):
        self.pi = np.random.beta(zeta/K, 1, size=(K))
        while np.any(self.pi == 0):
            self.pi[self.pi == 0] = np.random.beta(zeta[self.pi == 0], 1, size=(K))
        #print(pi)

    def generate_pi_2D(self,zeta,K):
        alpha_sk = np.tile(zeta / K,(K,1))
        T_num = 1
        for t in range(T_num):
            temp = np.transpose(np.random.beta(alpha_sk, 1))
            while np.any(temp == 0):
                temp[temp == 0] = np.transpose(np.random.beta(alpha_sk[temp == 0], 1))
            if t==0:
                self.pi = temp
            else:
                self.pi = self.pi+temp
        self.pi = self.pi/T_num

    # Here we are using binomial to generate random z
    # because z is bernouli ( binomial with only one element )
    def generate_Z(self,K,pi,pi_2D):
        self.Z = np.random.binomial( n=1, p=pi)
        while any(np.sum(self.Z,axis=1)==0):
            if pi_2D is True:
                self.Z[np.sum(self.Z, axis=1) == 0,] =  np.random.binomial(size=(sum(np.sum(self.Z, axis=1) == 0),K), n=1, p=pi[np.sum(self.Z, axis=1) == 0,])
            else:
                self.Z[np.sum(self.Z, axis=1) == 0,] = np.random.binomial(size=(sum(np.sum(self.Z, axis=1) == 0), K), n=1,p=pi)



    #self.Z = H > 0.05
    #self.Z = self.Z.astype(np.int)
    #print(z)
    def generate_G(self,F,S,K,F_epsilon,Z):
        self.G = np.zeros((S,K),dtype=np.float64)
        for s in range(S):
            try:
                alpha = np.power(F[:,0],Z[s][:])*np.power(F_epsilon[:,0],(1-Z[s][:]))
            except Exception:
                print("F or F_fraction has a problem. The exception is happened in generating G in the simulation.")
            scale = np.power(F[:,1],Z[s][:])*np.power(F_epsilon[:,1],(1-Z[s][:]))
            self.G[s][:] = np.random.gamma(shape=alpha,scale=scale,size=K)
        #print(G)

    def generate_H(self,G,S,K,Z):
        #self.H = G / sum(sum(G))
        self.H=np.zeros((S,K),dtype=np.float64)
        #G_modified = G*Z
        #self.H = G_modified/np.transpose(np.tile(G_modified.sum(axis=1),(K,1)))
        self.H = G / np.transpose(np.tile(G.sum(axis=1), (K, 1)))
        #for s in range(S):
         #   sum_temp = sum(G[s])
         #   self.H[s][:] = (G[s]/sum_temp)
        if np.sum(self.H==0)>0:
            temp = int(np.sum(self.H==0))
            print(G)
            print(self.F)
            print(self.H)
            raise Exception(str(temp)+" elements in H are zero in simulation")

    def generate_phi(self,r,q,I,K):
        self.phi = np.random.gamma(shape=r, scale=q, size=(I,K))
        if np.sum(self.phi==0):
            print(np.sum(self.phi==0))
            print("phi got zero in simulation of phi, fixing...")
            while np.sum(self.phi==0):
                print(np.sum(self.phi==0))
                self.phi[self.phi==0] = np.random.gamma(shape=r, scale=q, size=np.sum(self.phi==0))

    def generate_D(self,I,S,K,phi,H,n):
        self.D = np.zeros((I,S),dtype=np.int64)
        for i in range(I):
            for s in range(S):
                try:
                    self.D[i][s]=np.random.poisson(n[s]*sum(H[s][:]*phi[i][:]),size=1)
                except Exception:
                    print("problem in generating D")

    def generate_A(self,I,S,H,C,D,phi,theta):
        self.A = np.zeros((I,S),dtype=np.int64)
        self.p_binom = np.matmul(phi*C,np.transpose(H))/np.matmul(phi,np.transpose(H))
        self.p_binom = theta*self.p_binom
        try:
            self.A = np.random.binomial(n=D,p=self.p_binom)
        except Exception:
            print("problem in generating A in simulation class")

    def generate_B(self, b_alpha, b_beta, K, g):
        self.B = np.random.gamma(shape=b_alpha, scale=b_beta, size=(K, g))
        sums = np.sum(self.B, axis=0)
        while any(sums==0):
            self.b_alpha[sums==0] = np.random.gamma(self.b_alpha_shape, self.b_alpha_scale, size=sum(sums==0))
            self.B[:, sums==0] = np.random.gamma(shape=b_alpha[sums==0], scale=b_beta, size=(K, sum(sums==0)))
            sums = np.sum(self.B, axis=0)

    def generate_Y(self, n, H, B, p, S, g):
        Y = np.zeros((S, g))
        for gene in range(g):
            r = np.matmul(H, B[:, gene])
            r = np.multiply(r, n)
            r *= p[gene] / (1 - p[gene])
            Y_g = np.random.negative_binomial(r, p[gene], S)
            if np.sum(Y_g)==0:   # resampling - first try
                Y_g = np.random.negative_binomial(r, p[gene], S)
            while np.sum(Y_g)==0:   # didn't work -> resampling everything
                start = True
                while np.sum(self.B[:,gene]==0) or start:
                    self.b_alpha[gene] = np.random.gamma(self.b_alpha_shape, self.b_alpha_scale)
                    self.B[:,gene] = np.random.gamma(shape=self.b_alpha[gene], scale=self.b_beta, size=self.K)
                    start = False
                r = np.matmul(H, self.B[:, gene])
                r = np.multiply(r, n)
                r *= p[gene] / (1 - p[gene])
                Y_g = np.random.negative_binomial(r, p[gene], S)
            Y[:, gene] = Y_g
        self.Y = Y


