import numpy as np
import multiprocessing
import os
import scipy
import dfols
import time
from datetime import datetime
from functools import partial
import pickle 

class TTOptimizer:
    """"Optimizer for the TikTak algorithm."""

    def __init__(self,computation_options, global_search_options, local_search_options, skip_global=False):
        self.computation_options   = computation_options
        self.global_search_options = global_search_options
        self.local_search_options = local_search_options
        self.SRF = os.path.join(computation_options["working_dir"],"searchResults.dat")
        self.skip_global = skip_global


        if self.local_search_options['algorithm'].lower() == 'neldermead':
            self.minimizer = NelderMead
        elif self.local_search_options['algorithm'].lower() == 'dfols':
            self.minimizer = DFOLS
                
        else:
            raise RunTimeError("local search algorithm not recognized")


    def minimize(self,f,lower_bounds,upper_bounds,args=()):
        self.f =partial(f,*args)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.initial_step = 0.1*(upper_bounds-lower_bounds)
        self.nparam = len(lower_bounds)
        self.second_step_x=list()
        self.second_step_y=list()

        with open(self.SRF,"a") as srf:
            srf.write('----------------------------------' + '\n')
            srf.write('Begin logging optimization results' + '\n')
            srf.write(str(datetime.now()) + '\n')
            srf.write('----------------------------------' + '\n')
            

        self.mppool = multiprocessing.Pool(processes=self.computation_options["num_workers"])
        self.xstarts = list(self.GlobalSearch())
        best = self.LocalSearch()
        # print the last value of best
        print(f'best value {best[1]}')
        print(f'best point {best[0]}')
        
        #store the results
        order=np.argsort(self.second_step_y)
        with open('second_step.pkl', 'wb+') as file: pickle.dump([np.array(self.second_step_x)[order],np.array(self.second_step_y)[order]],file) 
            
        return best

    def GlobalSearch(self):

        if self.skip_global:
            
            #load the results
            with open('first_step.pkl', 'rb') as file:  
                xstarts=pickle.load(file)[1]
                       
        else:
            
            nsobol = self.global_search_options["num_points"]
                    
            #Generate sobol sequence  
            sampler = scipy.stats.qmc.Sobol(d=self.nparam, scramble=False) 
            sample_notscaled = sampler.random_base2(m=nsobol) 
            xstarts= scipy.stats.qmc.scale(sample_notscaled, self.lower_bounds,self.upper_bounds) 
       
            
            # --- evaluate f on many points ----
            aaa=np.array(self.mppool.map(self.f,xstarts))
            #xstarts[1,1,1,1,1,1]=4
            y = np.sum(aaa**2,axis=1)
            #xstarts[1,1,1,1]=4
            # sort the results
            I = np.argsort(y)
            xstarts = xstarts[I]
            y = y[I]
    
            #store the results
            with open('first_step.pkl', 'wb+') as file: pickle.dump([y,xstarts],file) 
        
        # take the best Imax
        xstarts = xstarts[:self.local_search_options["num_restarts"]]

        return xstarts


    def LocalSearch(self):
        """Function to manage the local searches.  Each local start is given to a consumer process that
        does the search.

        Parameters
        ~~~~~~~~~~~
        taskq       queue to submit tasks
        resultq     queue to receive results
        xstarts     the points at which we start in order
        best        best found so far (used if restarting after interuption )
        StartAt     how far into xstarts do we start  (used if restarting after interuption )
          """


        manager = multiprocessing.Manager()
        self.resultsq = manager.Queue()

        num_workers = self.computation_options["num_workers"]
        self.num_jobs = len(self.xstarts)




        self.best_so_far = (0.0, 1e10)


        self.result_trackers = []
        self.submit_counter = 0
        for i in range(num_workers):
            self.SubmitLocalResult()

        # the submissions happen recursively through the callback
        # here we wait until we have submitted all jobs
        while self.submit_counter < self.num_jobs:
            time.sleep(1)

        # here we wait until all jobs are done.
        [r.wait() for r in self.result_trackers]


        return self.best_so_far


    def SubmitLocalResult(self):
        i = self.submit_counter
        ishrink = self.local_search_options["shrink_after"]
        N= self.local_search_options["num_restarts"]
        newtask = self.xstarts.pop(0)
        
        
        if i >= ishrink: # shrink towards best so far
        
            theta = min(max(0.1,(i/N)**0.5),0.995)
            newtask = theta*self.best_so_far[0] + (1-theta)*newtask
        print(i,newtask)
        self.result_trackers.append(self.mppool.apply_async(localworker,
                (self.minimizer,
                (self.f,newtask,self.initial_step,self.lower_bounds,self.upper_bounds,self.local_search_options["xtol_rel"],self.local_search_options["ftol_rel"]),
                self.resultsq),
                callback = self.ProcessLocalResult,
                error_callback = self.ErrorCallback ) )

        self.submit_counter += 1


    def ProcessLocalResult(self,result):

        with open(self.SRF,"a") as srf:
            srf.write(str(result[1]) + ' ' + str(result[0]) + '\n')

        if result[1] < self.best_so_far[1] :
            print( 'best so far %f' % result[1])
            self.best_so_far = result

        if len(self.xstarts) > 0:
            self.SubmitLocalResult()
            
        self.second_step_x.append(result[0])  
        self.second_step_y.append(result[1])  

    def ErrorCallback(self,e):
        print('Error found in local search:')
        print(e)
        if len(self.xstarts) > 0:
            self.SubmitLocalResult()



def localworker(f,x,resultsq):
    answer = f(*x)
    resultsq.put( answer )
    return answer

def DFOLS(f,x,initial_step,lower_bounds,upper_bounds,xtol_rel,ftol_rel):
    
    
    res=dfols.solve(f,x, rhobeg = 0.1, rhoend=1e-5, maxfun=100, bounds=(np.array(lower_bounds),np.array(upper_bounds)), 
                npt=len(x)+5,scaling_within_bounds=True,  
                user_params={'tr_radius.gamma_dec':0.98,'tr_radius.gamma_inc':1.0, 
                              'tr_radius.alpha1':0.9,'tr_radius.alpha2':0.95}, 
                objfun_has_noise=False) 
        
   
    return (res.x, (np.array(res.f)**2).sum() )


def NelderMead(f,x,initial_step,lower_bounds,upper_bounds,xtol_rel,ftol_rel):
    """Note that NM doesn't impose bounds on optimization.
    The arguments initial_step, lower_bounds, and upper_bounds are not used.
    """
    res =  scipy.optimize.minimize(f, x, method='nelder-mead',
            options={'maxiter':100,'fatol': ftol_rel, 'xatol': xtol_rel, 'disp': False})
    return (res.x, res.fun)
