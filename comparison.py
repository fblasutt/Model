import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Bargaining_numba as brg


# plot style
linestyles = ['-','--','-.',':',':']
markers = ['o','s','D','*','P']
linewidth = 2
font_size = 8
font = {'size':font_size}
matplotlib.rc('font', **font)
plt.rcParams.update({'figure.max_open_warning': 0,'text.usetex': False})


# settings for models to solve

xc=np.array([0.59909138, 0.02361305, 0.03624229, 0.75706021, 0.48123509])

specs = {'model 1':{'latexname':'EGM2', 'par':{'θ':xc[0],'meet':1.0,'σL0':xc[1],'σL':xc[2],'α2':xc[3],'α1':1.0-xc[3],'γ':xc[4]}}}
root='C:/Users/32489/Dropbox/Family Risk Sharing'



# solve different models (takes several minutes)
models = {}
for name,spec in specs.items():
    print(f'solving {name}...')
    
    # setup model
    models[name] = brg.HouseholdModelClass(name=name,par=spec['par'])
    models[name].spec = spec
    
    # solve
    models[name].solve()
    
   
        
#Policy Functions
cmaps = ('viridis','gray')
model_list = ('model 1',)

#Points to consider
t = 0; iz=0; ih=0

par = models['model 1'].par
for iL in (par.num_love//2,): 
    for var in ('p_Vw_remain_couple','n_C_tot_remain_couple','power','remain_WLP'):

        fig = plt.figure();ax = plt.axes(projection='3d')
                
        for i,name in enumerate(model_list):
            model = models[name]
            par = models[name].par
            X, Y = np.meshgrid(par.grid_power, par.grid_A,indexing='ij')
            
            Z = getattr(model.sol,var)[t,ih,iz,:,iL,:]
            alpha = 0.2 if name=='model 1' else 0.5
            ax.plot_surface(X, Y, Z,cmap=cmaps[i],alpha=alpha);
            if var == 'power':  ax.set(zlim=[0.0,1.0])
            ax.set(xlabel='power',ylabel='$A$');ax.set_title(f'{var}')
    
# Simulated Path
var_list = ('couple','A','power','love','WLP')
model_list = ('model 1',)
init_power=model.par.grid_power[0];init_love=par.num_love//2
for i,name in enumerate(model_list):
    model = models[name]

    # show how starting of in a low bargaining power gradually improves
    model.sim.init_power[:] = init_power
    model.sim.init_love[:] = init_love 
    model.simulate()
    
for var in var_list:

    fig, ax = plt.subplots()
    
    for i,name in enumerate(model_list):
        model = models[name]

        # pick out couples (if not the share of couples is plotted)
        if var == 'couple': nan = 0.0
        else:
            I = model.sim.couple<1
            nan = np.zeros(I.shape)
            nan[I] = np.nan

        # pick relevant variable for couples
        y = getattr(model.sim,var);y = np.nanmean(y + nan,axis=0)
        ax.plot(y,marker=markers[i],linestyle=linestyles[i],linewidth=linewidth,label=model.spec['latexname']);
        ax.set(xlabel='age',ylabel=f'{var}');ax.set_title(f'pow_idx={init_power}, init_love={init_love}')


#############################################################
#Risk of rebargaining, match quality, and relative earnings
##############################################################
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


M=model

age=(np.cumsum(np.ones((M.par.simN,M.par.simT)),axis=1)-1)
sampl = (age<=35) & (age>=1) & (M.sim.couple_lag==1) #&  (M.sim.couple==1)
sampl2 = (age<=35) & (age>=1) & (M.sim.couple_lag==1) &  (M.sim.couple==1)
lov,rel,plov,prel=np.zeros((4,model.par.simN,M.par.T))

for i in range(M.par.T):lov[:,i]=M.par.grid_love[i][M.sim.love[:,i]]
rel=M.sim.incw/M.sim.incm


Sw=M.sim.Vcw-M.sim.Vsw
Sm=M.sim.Vcm-M.sim.Vsm
     






def heatmap(x,y,z,bins_x,bins_y,tick_density,Xlabel,Ylabel,Zlabel,fig_name):

    # Create 2D histogram (binning x and y, aggregating z as mean)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[bins_x, bins_y], weights=z)
    heatmap_counts, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y])  # Count per bin
    
    # Avoid division by zero by replacing zero counts with NaN
    heatmap_counts[heatmap_counts == 0] = np.nan
    heatmap = heatmap / heatmap_counts  # Compute mean value in each bin
    
    # Reverse the y-axis data for a bottom-to-top y-axis
    heatmap = heatmap[:, ::-1]
    yticks = np.round(yedges[:-1], 1)[::-1]  # Reverse y-tick labels
    
    # Create a heatmap with Seaborn
    sns.heatmap(
        heatmap.T,  # Transpose to align axes properly
        xticklabels=np.round(xedges[:-1], 1),  # Use bin edges as labels
        yticklabels=yticks,  # Reversed y-tick labels for correct ordering
        cmap='viridis',  # Customize color palette
        cbar_kws={'label': Zlabel}  # Add color bar label
    )
    
    # Reduce tick density
    plt.xticks(ticks=np.arange(0, len(xedges[:-1]), tick_density), labels=np.round(xedges[::tick_density], 1))  # Show fewer x ticks
    plt.yticks(ticks=np.arange(0, len(yedges[:-1]), tick_density), labels=np.round(yedges[::tick_density], 1)[::-1])  # Show fewer y ticks
    
    # Add labels and title
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    
    #Save the picture
    plt.savefig(fig_name, format='eps', bbox_inches="tight")  
    # Show the plot
    plt.show()
    
heatmap(lov[sampl],#x axis
        rel[sampl],#y axis
        M.sim.power[sampl]!=M.sim.power_lag[sampl],#Z axis   
        np.linspace(-0.35, 0.35, 20),#X bins
        np.linspace(0.0, 2, 20),#Y bins
        4,#1/(ticks density)
        'Match quality','Earnins w/Earnings m','Share renegotiations or divorces',
        root+'/Model/results/match_earn_ren_div.eps')



heatmap(np.roll(Sw,1)[sampl],#x axis
        np.roll(Sm,1)[sampl],#y axis
        M.sim.power[sampl]!=M.sim.power_lag[sampl],#Z axis   
        np.linspace(-1.0, 3, 100),  # 10 bins for X
        np.linspace(-1.0, 3, 100),  # 10 bins for Y
        6,#1/(ticks density)
        'Surplus W','Surplus M','Share renegotiations or divorces',
        root+'/Model/results/surp_ren_div.eps')

heatmap(np.roll(Sw,1)[sampl],#x axis
        np.roll(Sm,1)[sampl],#y axis
        M.sim.power[sampl]==-1,#Z axis   
        np.linspace(-1.0, 3, 100),  # 10 bins for X
        np.linspace(-1.0, 3, 100),  # 10 bins for Y
        6,#1/(ticks density)
        'Surplus W','Surplus M','Share renegotiations or divorces',
        root+'/Model/results/surp_div.eps')


heatmap(np.roll(Sw,1)[sampl2],#x axis
        np.roll(Sm,1)[sampl2],#y axis
        M.sim.power[sampl2]<M.sim.power_lag[sampl2],#Z axis   
        np.linspace(-1.0, 3, 100),  # 10 bins for X
        np.linspace(-1.0, 3, 100),  # 10 bins for Y
        6,#1/(ticks density)
        'Surplus W','Surplus M','Share renegotiations triggered by w',
        root+'/Model/results/match_renw.eps')

heatmap(np.roll(Sw,1)[sampl2],#x axis
        np.roll(Sm,1)[sampl2],#y axis
        M.sim.power[sampl2]>M.sim.power_lag[sampl2],#Z axis   
        np.linspace(-1.0, 3, 100),  # 10 bins for X
        np.linspace(-1.0, 3, 100),  # 10 bins for Y
        6,#1/(ticks density)
        'Surplus W','Surplus M','Share renegotiations triggered by m',
        root+'/Model/results/match_renm.eps')


#Distribution of marital surplus
fig, axs = plt.subplots(nrows=1, ncols=1)
axs.hist2d(Sw[sampl], Sm[sampl], bins=30,
           range=[[0.0,5.0], [0.0, 5]])
axs.set(xlabel='Surplus, w',ylabel='Surplus, m')
fig.tight_layout()
plt.savefig(root+'/Model/results/surplus_dist.eps', format='eps', bbox_inches="tight")    
plt.show()


#Life-cycle behavior

#Aggregate
keep=M.sim.couple<=1
plt.plot(np.mean(M.sim.xw,axis=0,where=keep),label="Public exp.")
plt.plot(np.mean(M.sim.A,axis=0,where=keep),label="Total Assets")
plt.plot(np.mean(M.sim.incm+M.sim.incw*M.sim.WLP,axis=0,where=keep),label="Total earnings")
plt.plot(np.mean(M.sim.Cw,axis=0,where=keep),label="Priv C, w")
plt.plot(np.mean(M.sim.Cm,axis=0,where=keep),label="Priv C, m")
plt.legend()
plt.xlabel("Age") 
plt.ylim(0, 20)       
plt.savefig(root+'/Model/results/lifecycle_aggregate.eps', format='eps', bbox_inches="tight")                   
plt.show()
         
#Married
keep=M.sim.couple==1
plt.plot(np.mean(M.sim.xw,axis=0,where=keep),label="Public exp.")
plt.plot(np.mean(M.sim.A,axis=0,where=keep),label="HH Assets")
plt.plot(np.mean(M.sim.incm+M.sim.incw*M.sim.WLP,axis=0,where=keep),label="HH earnings")
plt.plot(np.mean(M.sim.Cw,axis=0,where=keep),label="Priv C, w")
plt.plot(np.mean(M.sim.Cm,axis=0,where=keep),label="Priv C, m")
plt.xlabel("Age") 
plt.ylim(0, 20)       
plt.legend()      
plt.savefig(root+'/Model/results/lifecycle_married.eps', format='eps', bbox_inches="tight")                          
plt.show()

#Single
keep=M.sim.couple==0
plt.plot(np.mean(M.sim.xw,axis=0,where=keep),label="Public exp.")
plt.plot(np.mean(M.sim.Aw,axis=0,where=keep),label="Assets")
plt.plot(np.mean(M.sim.incm+M.sim.incw*M.sim.WLP,axis=0,where=keep),label="Earnings, w")
plt.plot(np.mean(M.sim.Cw,axis=0,where=keep),label="Priv C, w")
plt.xlabel("Age")    
plt.ylim(0, 20)    
plt.legend()                              
plt.savefig(root+'/Model/results/lifecycle_singlew.eps', format='eps', bbox_inches="tight")  
plt.show()



    
#####################################################
#Descriptives about assets, earnings, and consumption
#####################################################
def ginii(x):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    sorted_x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(sorted_x, dtype=float)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def tops(x,p=1):
    top1=np.percentile(x,100-p)
    return np.sum(x[x>top1])/np.sum(x)

assets=M.sim.A*(M.sim.couple==1)+M.sim.Aw*(M.sim.couple==0)
earnings=(M.sim.incm+M.sim.incw*M.sim.WLP)*(M.sim.couple==1)+(M.sim.incw*M.sim.WLP)*(M.sim.couple==0)


list_a=[assets[M.sim.couple<=1],earnings[M.sim.couple<=1],M.sim.Cw[M.sim.couple==1],M.sim.Cm[M.sim.couple==1],M.sim.xw[M.sim.couple==1]]
list_s=['A','E','Cw','Cm','X']

mean={s:i.mean()  for i,s in zip(list_a,list_s)}
top1={s:tops(i)   for i,s in zip(list_a,list_s)}
gini={s:ginii(i)  for i,s in zip(list_a,list_s)}

def p33(x): y=x;return str('%3.3f' % y)     
 
#Table with summary stats
table=r'Mean          & '+p33(mean['A'])+' & '+p33(mean['E'])+' & '+p33(mean['Cw'])+' & '+p33(mean['Cm'])+' & '+p33(mean['X'])+'    \\\\ '+\
      r'Gini          & '+p33(gini['A'])+' & '+p33(gini['E'])+' & '+p33(gini['Cw'])+' & '+p33(gini['Cm'])+' & '+p33(gini['X'])+'    \\\\ '+\
      r'Top 1\% share & '+p33(top1['A'])+' & '+p33(top1['E'])+' & '+p33(top1['Cw'])+' & '+p33(top1['Cm'])+' & '+p33(top1['X'])+'    \\\\\\bottomrule'
with open(root+'/Model/results/sum_stat.tex', 'w') as f: f.write(table); f.close() 

    




