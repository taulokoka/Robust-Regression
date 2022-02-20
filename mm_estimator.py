import numpy as np
import scipy as sp
import math
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class MMEstimator(BaseEstimator, RegressorMixin):
    """
    MM-estimator of regression initialized with S-estimate with high breakingpoint.
    The latter is computed according to [1]. The Code is a basic translation from the Matlab
    implementation in [2] into Python. 
    (Translated to Python by Taulant Koka, TU Darmstadt, Germany)

    [1]  "A fast algorithm for S-regression estimates.",Salibian-Barrera, M. and Yohai, V. (2005),
    [2]  https://feb.kuleuven.be/public/u0017833/Programs/mme/MMrse.txt 

    """
    def __init__(self, bdp_S=0.5,bdp_mm=0,N=20,k=2,bestr=5,initialscale=None):
        """Initialize Estimator with parameters:

        Args:
            bdp_S (float, optional): Breakdownpoint of the S-Estimating step. Defaults to 0.5.
            bdp_mm (int, optional): Breakdownpoint of MM-Estimating step; at bdp=0 the Estimator has 
                                    0.95 efficiency for normal distribution. Defaults to 0.
            N (integer, optional): Number of subsamples. Defaults to 20.
            k (integer, optional): Number of iterations for the refining per subsample. Defaults to 2.
            bestr (integer, optional): Number of best coefficients to iterate through. Defaults to 5.
            initialscale (float, optional): Use "initialscale" if present, otherwise MAD is used as initialization
                                    for iteratively reweighted least squares (IRWLS). Defaults to None.
        """
        self.bdp_S = bdp_S
        self.bdp_mm = bdp_mm
        self.N = N
        self.k = k
        self.bestr = bestr
        self.initialscale = initialscale

    def fit(self, X, y):

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.X_ = X
        self.y_ = y
        beta,sigma = self.MMrse(self.y_, self.X_)
        self.coef_ = beta
        self.sigma_ = sigma
        # Return the classifier
        return self

    def predict(self, X):

        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        predicted = X@self.coef_
        
        return predicted

    def MMrse(self,y_in, X_in):
        """
        Computes MM-estimator of regression S-estimate with high breakingpoint as initialization 
        The latter is computed according to. 
  
        Args:
            y_in (ndarray): 1D array containing measurement of length N
            X_in (ndarray): 2D array containing the design matrix 
            bdp (float): Breakdown point (e.g. 0.25 or 0.50)

        Returns:
            beta (ndarray): 1D array containing estimated regression coefficients
            sigma (float): Estimated regression scale
        """
        Y = y_in.copy()
        X = X_in.copy()
        bs,ss = self.fastsreg(X,Y)
        bm = self.mmregres(X,Y,bs,ss)
        beta = bm
        sigma = ss
        return beta,sigma

    def fastsreg(self, x, y):
        """
        Fast S-Estimatior implemented according to [1] (translated from [2]).
        An implementation in R can be found in [3] 
        
        [1]  "A fast algorithm for S-regression estimates.",Salibian-Barrera, M. and Yohai, V. (2005),
        [2]  https://feb.kuleuven.be/public/u0017833/Programs/mme/MMrse.txt 
        [3]  http://hajek.stat.ubc.ca/~matias/soft.html
        Args:
            x (ndarray): 2D array containing the data matrix with dimensions (n x p)
            y (ndarray): 1D array containing the measurements (n, )
        Returns:
            beta (ndarray): Robust estimate of the regression coefficients (p, )
            scale (float): Values of the objective function
        """

        n,p =np.shape(x)
        c = self.Tbsc(self.bdp_S,1)
        kp = (c/6) * self.Tbsb(c,1)

        bestbetas = np.zeros((self.bestr,p))
        bestscales = 1e20 * np.ones(self.bestr)
        sworst = 1e20
            
        for i in range(self.N):
            # get subsample
            singular =1 
            itertest=1
            while (np.sum(singular)>=1) and (itertest<100):
                index = np.random.permutation(np.arange(n))
                index=index[:p]
                xs = x[index,:]
                ys = y[index]
                beta = self.oursolve(xs,ys)
                singular = np.isnan(beta)
                itertest=itertest+1
            if itertest==100:
                print('too many degenerate subsamples') 
            if self.k>0 :
            # refine
                res,betarw,scalerw = self.ress(x,y,beta,self.k,0,kp,c)
                betarw = betarw
                scalerw = scalerw
                resrw = res
            else:
            # "no refining"
                betarw = beta
                resrw = y - x @ betarw
                scalerw = np.median(np.absolute(resrw))/.6745
            if i > 1:
                scaletest = self.lossS(resrw,sworst,c)
                if scaletest < kp:
                    sbest = self.scale1(resrw,kp,c,scalerw)
                    yi= np.argsort(bestscales)
                    ind=yi[self.bestr-1]
                    bestscales[ind] = sbest
                    bestbetas[ind,:] = betarw.T
                    sworst = max(bestscales)
            else:
                bestscales[self.bestr-1] = self.scale1(resrw,kp,c,scalerw)
                bestbetas[self.bestr-1,:] = betarw.T

        # refine until convergence, starting from best candidate
        superbestscale = 1e20
        for i in range(self.bestr-1,1,-1):
            self.initialscale = bestscales[i]
            _,betarw,scalerw = self.ress(x,y,bestbetas[i,:].T,0,1,kp,c)
            if scalerw < superbestscale:
                superbestscale = scalerw
                superbestbeta = betarw
        beta=superbestbeta
        scale=superbestscale
        return beta,scale

    def mmregres(self,X,Y,b0,s):
        """ 
        Args:
            X (ndarray): 2D array containing data matrix (n x p)
            Y (ndarray): 1D array containing measurements (n, )
            b0 (ndarray): 1D array containing initial S-estimate of regression coefficients (p, ) 
            s (float): Estimated regression scale

        Returns:
            beta_mm (ndarray): 1D array containing the MM-estimate of regression coefficients (p, )
        """
    
        k=min(np.shape(X))
        if self.bdp_mm == 0:
            c=4.685    
        else:
            c=self.Tbsc(self.bdp_mm,1)
        maxit=100;tol=10**(-10)
        eps=10**(-200)
        iter=0;crit=1000
        b1=b0
        while (iter <= maxit) and (crit > tol):  
            r1=(Y-X@b1)/s
            tmp = np.nonzero(abs(r1) <= eps)  
            n1,n2 = np.shape(tmp)
            if n1 != 0: 
                r1[tmp] = eps   
            w = self.psibi(r1,c)/r1
            W = np.diag(w)@np.ones((len(w),k))
            XW = X.T * W.T
            b2= np.linalg.pinv(XW@X)@XW@Y
            d=b2-b1
            crit = max(np.absolute(d))
            iter=iter+1
            b1=b2
        beta_mm=b2
        return beta_mm

    def dpsibi(self,x,c):
        '''
        computes derivative of tukey's biweight psi function with constant c for 
        all values in the vector x.
        '''
        z = (abs(x) < c) * (1 - x**2 *(6/c**2 - 5*x**2/c**4))
        return z

    def fw(self,u,c):
        '''
        weight function = psi(u)/u

        '''
        tmp = (1 - (u/c)**2)**2
        tmp = tmp * (c**2/6)
        tmp[abs(u/c) > 1] = 0
        return tmp

    def gint(self,k,c,p):
        '''
        Integral from zero to c of r^k g(r^2), where g(||x||^2) is the density function
        of a p-dimensional standardnormal distribution 
        '''
        e=(k-p-1)/2
        numerator=(2**e)*sp.stats.gamma.cdf((c**2)/2,(k+1)/2)*math.gamma((k+1)/2)
        return numerator/(np.pi**(p/2))


    def lossS(self,u,s,c):
        return np.mean(self.rhobi(u/s,c))

    def oursolve(self,X,y):
        '''
        Solve linear equation
        '''
        p = np.shape(X)[1]
        if np.linalg.matrix_rank(X) < p:
            beta_est = np.nan
        else: 
            beta_est = np.linalg.pinv(X) @ y
        return beta_est

    def psibi(self,x,c):
        '''
        psi function for biweight. 
        c : tuning parameter
        '''
        z = (abs(x) < c) * x * ( 1 - (x/c)**2 )**2
        return z

    def rhobi(self,u,c):
        '''
        rho function for biweight.  Used for robust scale estimation.
        c: tuning parameter
        '''
        w = (np.absolute(u)<=c)
        return (u**2/(2)*(1-(u**2/(c**2))+(u**4/(3*c**4))))*w +(1-w)*(c*2/6)

    def ress(self,x,y,initialbeta,k,conv,kp,c):
        '''
        Perform Iteratively reweighted least squares (IRWLS) k times to refine from
        initial beta

        Use "initialscale" if present, otherwise MAD is used as initialization
        k = number of refining steps
        conv = 0 means "do k steps and don't check for convergence"
        conv = 1 means "stop when convergence is detected, or the maximum number
                        of iterations is achieved"
        kp and c = tuning constants of the equation
        '''
        n,p=np.shape(x)    
        res = y - x @ initialbeta
        if self.initialscale==None:
            scale = np.median(np.absolute(res))/.6745
            self.initialscale = scale
        else:
            scale = self.initialscale
        
        if conv == 1: 
            k = 50

        beta = initialbeta
        scale = self.initialscale
        for i in range(k):
            scale = np.sqrt( scale**2 * np.mean(self.rhobi(res/scale,c) ) / kp )
            # IRWLS one step
            weights = self.fw(res/scale,c)
            sqweights = weights**(1/2)
            sqW = np.diag(sqweights)@np.ones((n,p))
            xw = x * sqW
            yw = y * sqweights
            beta1 = self.oursolve(xw.T@xw,xw.T@yw)
            if (np.isnan(beta1).any()):
                beta1 = initialbeta
                scale = self.initialscale
                break
            
            if  conv==1:
                if ( np.linalg.norm( beta - beta1 )/np.linalg.norm(beta) < 1e-20 ): 
                    break
                res = y - x @ beta1
            beta = beta1
        res = y - x @ beta
        return res,beta1,scale

    def scale1(self,u, kp, c, initialsc=None):
        '''
        Compute scale
        '''
        if initialsc == None:
            initialsc = np.median(abs(u))/0.6745
        maxit = 200
        sc = initialsc
        i = 0
        eps = 1e-20
        err = 1
        while  (( i < maxit ) and (err > eps)):
            sc2 = np.sqrt( sc**2 * np.mean(self.rhobi(u/sc,c)) / kp)
            err =abs(sc2/sc - 1)
            sc = sc2
            i=i+1
        return sc


    def Tbsb(self,c,p):
        y1  =   self.gint(p+1,c,p)/2 - self.gint(p+3,c,p)/(2*c**2) + self.gint(p+5,c,p)/(6*c**4)
        y2  =   (6/c)*2*(np.pi**(p/2))/math.gamma(p/2)
        y3  =   c*(1-sp.stats.chi2.cdf(c**2,p))
        return y1*y2+y3

    def Tbsc(self,alpha,p):
        '''
        constant for Tukey Biweight S
        ''' 
        talpha = np.sqrt(sp.stats.chi2.ppf(1-alpha,p))
        maxit = 1000
        eps = 10**(-8)
        diff = 10**6
        ctest = talpha
        iter = 1

        while ((diff>eps) and iter<maxit):
            cold = ctest
            ctest = self.Tbsb(cold,p)/alpha
            diff = abs(cold-ctest)
            iter = iter+1
        return ctest
