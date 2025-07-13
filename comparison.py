import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import Bargaining_numba as brg
import pandas as pd
from scipy.stats import kurtosis,skew

# plot style
linestyles = ['-','--','-.',':',':']
markers = ['o','s','D','*','P']
linewidth = 2
font_size = 8
font = {'size':font_size}
matplotlib.rc('font', **font)
plt.rcParams.update({'figure.max_open_warning': 0,'text.usetex': False})


# settings for models to solve

#Initialize seed 
np.random.seed(10) 
 
#Create sample with replacement 
N=10_000#sample size 

#Root
root='C:/Users/32489/Dropbox/Family Risk Sharing'


baseline_sample=np.array(pd.read_excel(root+'/Output files/data_sample.csv'))

pr=np.ones(baseline_sample.shape[0])/baseline_sample.shape[0]
indexes=np.array(np.random.choice(baseline_sample[:,0], size=N, p=pr, replace=True),dtype=np.int32)-1
final_sample= baseline_sample[:,1:][indexes] 

age_initial=final_sample[:,0]
age_final=final_sample[:,1]
cw_cons_share=final_sample[:,2]
h_income=final_sample[:,3]
w_income=final_sample[:,4]
age_marriage=final_sample[:,5]

# POints: [ν,σL,α,χ,wedge]
xc=np.array([0.37139106, 0.00931249, 0.90003228, 1.28494861, 0.81349789])

#Parametrize the model 
par = {'simN':N,'ν': xc[0],'σL':xc[1],'α':xc[2],'χ':xc[3],'wedge':xc[4]} 
model = brg.HouseholdModelClass(par=par)  

###########################################################
# Set the initial conditions for the couples
###########################################################

#We start simulating the agent at age_initial
model.par.sample_init=age_initial-20

#Given the parameters, set the initial pareto weight for couples
param=(cw_cons_share/(1.0-cw_cons_share))**model.par.ρ
model.sim.init_power=param/(1.0+param)

#Set the initial income gridpoints for income, the closest to our value
izm=np.array([np.argmin(np.abs(np.log(model.par.grid_zm)[int(model.par.sample_init[i]),:,0]-h_income[i])) for i in range(model.par.simN)],dtype=np.int32)
izm[np.isnan(h_income)]=(model.par.num_pm*model.par.num_ϵm)//2


izw=np.array([np.argmin(np.abs(np.log(model.par.grid_zw)[int(model.par.sample_init[i]),:,0]-w_income[i])) for i in range(model.par.simN)],dtype=np.int32)
izw[np.isnan(w_income)]=(model.par.num_pw*model.par.num_ϵw)//2

        
model.sim.init_z=izm*model.par.num_zm+izw


# solve different models (takes several minutes)
model.solve()
model.simulate()
    
   
        
#Policy Functions
cmaps = ('viridis','gray')
model_list = ('model 1',)

#Points to consider
t = 0; iz=0; ih=0;wls=1

par = model.par
for iL in (par.num_love//2,): 
    for var in ('i_Vw_remain_couple','i_C_tot_remain_couple','remain_WLP'):

        fig = plt.figure();ax = plt.axes(projection='3d')
                
        for i,name in enumerate(model_list):

            X, Y = np.meshgrid(par.grid_power, par.grid_A,indexing='ij')
            
            Z = getattr(model.sol,var)[t,wls,ih,iz,:,iL,:]
            alpha = 0.2 if name=='model 1' else 0.5
            ax.plot_surface(X, Y, Z,cmap=cmaps[i],alpha=alpha);
            if var == 'power':  ax.set(zlim=[0.0,1.0])
            ax.set(xlabel='power',ylabel='$A$');ax.set_title(f'{var}')
    
# Simulated Path
var_list = ('couple','A','power','love','WLP')
model_list = ('model 1',)


    
for var in var_list:

    fig, ax = plt.subplots()
    
    for i,name in enumerate(model_list):

        # pick out couples (if not the share of couples is plotted)
        if var == 'couple': nan = 0.0
        else:
            I = model.sim.couple<1
            nan = np.zeros(I.shape)
            nan[I] = np.nan

        # pick relevant variable for couples
        y = getattr(model.sim,var);y = np.nanmean(y + nan,axis=0)
        ax.plot(y,marker=markers[i],linestyle=linestyles[i],linewidth=linewidth);
        ax.set(xlabel='age',ylabel=f'{var}');


#############################################################
#Risk of rebargaining, match quality, and relative earnings
##############################################################
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


M=model


age=(np.cumsum(np.ones((M.par.simN,M.par.T)),axis=1)-1)+20#age of hh   
sampl =  (age>=age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple_lag==1)
sampl2 =(sampl) &  (M.sim.couple==1)
lov,rel,plov,prel=np.zeros((4,model.par.simN,M.par.T))

for i in range(M.par.T):lov[:,i]=M.par.grid_love[i][M.sim.love[:,i]]
rel=M.sim.incw/M.sim.incm


#Income and consumption growth

ΔYm  =np.log(M.sim.incm)  -np.log(np.roll(M.sim.incm,1,axis=1))
ΔYw  =np.log(M.sim.incw)  -np.log(np.roll(M.sim.incw,1,axis=1))

ΔCm  =np.log(M.sim.Cm)  -np.log(np.roll(M.sim.Cm,1,axis=1))
ΔCw  =np.log(M.sim.Cw)  -np.log(np.roll(M.sim.Cw,1,axis=1))
Δws =np.log(M.sim.Cw/(M.sim.Cm))  -np.log(np.roll(M.sim.Cw/(M.sim.Cm),1,axis=1))#poww[:,b+1:e+1]-poww[:,b:e]#np.log(Q[:,b+1:e+1]/(m.sim.C_tot[:,b+1:e+1]))  -np.log(Q[:,b:e]/(m.sim.C_tot[:,b:e]))#
ΔC =np.log(M.sim.C_tot)  -np.log(np.roll(M.sim.C_tot,1,axis=1))
Δd=np.log(M.sim.dw)  -np.log(np.roll(M.sim.dw,1,axis=1))
    


Sw=M.sim.Vcw-M.sim.Vsw
Sm=M.sim.Vcm-M.sim.Vsm
     






def heatmap(x,y,z,bins_x,bins_y,tick_density,Xlabel,Ylabel,Zlabel,fig_name,vmax=1.0):

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
        vmax=vmax,
        cbar_kws={'label': Zlabel}  # Add color bar label
    )
    
    # Reduce tick density
    plt.xticks(ticks=np.arange(0, len(xedges[:-1]), tick_density), labels=np.round(xedges[::tick_density], 1))  # Show fewer x ticks
    plt.yticks(ticks=np.arange(0, len(yedges[:-1]), tick_density), labels=np.round(yedges[::tick_density], 1)[::-1])  # Show fewer y ticks
    
    # Add labels and title
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)
    
    #Add the overall average
    meanz=z.mean()
    plt.text(-0.9, -0.9,  f'Avg. is {meanz:.4f}', size=10)
    
    #Save the picture
    plt.savefig(fig_name, format='eps', bbox_inches="tight")  
    # Show the plot
    plt.show()
    
#Relative earnings, match quality, divorces and renegotiations
heatmap(lov[sampl],#x axis
        rel[sampl],#y axis
        (M.sim.power[sampl]!=M.sim.power_lag[sampl]),#Z axis   
        np.linspace(-0.35, 0.35, 20),#X bins
        np.linspace(0.0, 2, 20),#Y bins
        4,#1/(ticks density)
        'Match quality','Earnins w/Earnings m','Share renegotiations or divorces',
        root+'/Model/results/match_earn_ren_div.eps')


#Men, women's earnings and renegotiations
heatmap(ΔYm[sampl],#x axis
        ΔYw[sampl],#y axis
        (M.sim.power[sampl]!=M.sim.power_lag[sampl]),#Z axis   
        np.linspace(0, 1.5, 10),#X bins
        np.linspace(0, 1.5, 10),#X bins
        2,#1/(ticks density)
        'M income shock','W income shock','Share renegotiation or divorced',
        root+'/Model/results/shocks_ren_div.eps',
        vmax=0.2)#max value displayed

heatmap(ΔYm[sampl],#x axis
        ΔYw[sampl],#y axis
        (M.sim.power[sampl]<0),#Z axis   
        np.linspace(0, 1.5, 10),#X bins
        np.linspace(0, 1.5, 10),#X bins
        2,#1/(ticks density)
        'M income shock','W income shock','Share divorces',
        root+'/Model/results/shocks_div.eps',
        vmax=0.2)#max value displayed)

heatmap(ΔYm[sampl],#x axis
        ΔYw[sampl],#y axis
        (M.sim.power[sampl]>M.sim.power_lag[sampl]) & (M.sim.power[sampl]>0),#Z axis   
        np.linspace(0, 1.5, 10),#X bins
        np.linspace(0, 1.5, 10),#X bins
        2,#1/(ticks density)
        'M income shock','W income shock','Share renegotiation by W',
        root+'/Model/results/shocks_ren_w.eps',
        vmax=0.2)#max value displayed


heatmap(ΔYm[sampl],#x axis
        ΔYw[sampl],#y axis
        (M.sim.power[sampl]<M.sim.power_lag[sampl]) & (M.sim.power[sampl]>0),#Z axis   
        np.linspace(0, 1.5, 10),#X bins
        np.linspace(0, 1.5, 10),#X bins
        2,#1/(ticks density)
        'M income shock','W income shock','Share renegotiation by M',
        root+'/Model/results/shocks_ren_m.eps',
        vmax=0.2)#max value displayed





#Men, women's surplus and renegotiations
heatmap(np.roll(Sw,1)[sampl],#x axis
        np.roll(Sm,1)[sampl],#y axis
        M.sim.power[sampl]!=M.sim.power_lag[sampl],#Z axis   
        np.linspace(0, 3, 20),  # 10 bins for X
        np.linspace(0, 3, 20),  # 10 bins for Y
        6,#1/(ticks density)
        'Surplus W','Surplus M','Share renegotiations or divorces',
        root+'/Model/results/surp_ren_div.eps',
        vmax=0.6)#max value displayed

heatmap(np.roll(Sw,1)[sampl],#x axis
        np.roll(Sm,1)[sampl],#y axis
        M.sim.power[sampl]<0,#Z axis   
        np.linspace(0, 3, 20),  # 10 bins for X
        np.linspace(0, 3, 20),  # 10 bins for Y
        6,#1/(ticks density)
        'Surplus W','Surplus M','Share divorces',
        root+'/Model/results/surp_div.eps',
        vmax=0.6)#max value displayed


heatmap(np.roll(Sw,1)[sampl],#x axis
        np.roll(Sm,1)[sampl],#y axis
        (M.sim.power[sampl]>M.sim.power_lag[sampl]) & (M.sim.power[sampl]>0),#Z axis   
        np.linspace(0, 3, 20),  # 10 bins for X
        np.linspace(0, 3, 20),  # 10 bins for Y
        6,#1/(ticks density)
        'Surplus W','Surplus M','Share renegotiations triggered by w',
        root+'/Model/results/surp_renw.eps',
        vmax=0.6)#max value displayed

heatmap(np.roll(Sw,1)[sampl],#x axis
        np.roll(Sm,1)[sampl],#y axis
        (M.sim.power[sampl]<M.sim.power_lag[sampl]) & (M.sim.power[sampl]>0),#Z axis  
        np.linspace(0, 3, 20),  # 10 bins for X
        np.linspace(0, 3, 20),  # 10 bins for Y
        6,#1/(ticks density)
        'Surplus W','Surplus M','Share renegotiations triggered by m',
        root+'/Model/results/surp_renm.eps',
        vmax=0.6)#max value displayed


#Distribution of marital surplus
fig, axs = plt.subplots(nrows=1, ncols=1)
axs.hist2d(Sw[sampl], Sm[sampl], bins=20,
           range=[[0.0,3.0], [0.0, 3]],vmax=np.sum(Sw[sampl]>=0)/200,vmin=0.0)
axs.set(xlabel='Surplus, w',ylabel='Surplus, m')
fig.tight_layout()
plt.savefig(root+'/Model/results/surplus_dist.eps', format='eps', bbox_inches="tight")    
plt.show()

#Distribution of marital surplus
sns.histplot(1e-2+Sw[:,:M.par.Tr].flatten(),log_scale=True,  stat='percent', label='Wife surplus',color='red')
sns.histplot(1e-2+Sm[:,:M.par.Tr].flatten(),log_scale=True,  stat='percent', label='Husband surplus',color='blue')
plt.legend()
plt.savefig(root+'/Model/results/surplus_dist_2d.eps', format='eps', bbox_inches="tight")  
plt.show()


sns.histplot(Sm[(M.sim.couple_lag==0)].flatten(), log_scale=True , stat='percent',  label='Husband surplus',color='blue')
sns.histplot(Sw[(M.sim.couple_lag==0)].flatten(), log_scale=True,  stat='percent',  label='Wife surplus',color='red')
plt.legend()
plt.savefig(root+'/Model/results/surplus_dist_i_2d.eps', format='eps', bbox_inches="tight")  
plt.show()


###########################################
#Distribution of log  consumption growth
##########################################

sns.displot(ΔCw[(M.sim.couple_lag==0)].flatten(),color='blue',kind="kde")
sns.displot(ΔCm[(M.sim.couple_lag==0)].flatten(),color='red',kind="kde")
#sns.histplot(ΔYm[(M.sim.couple_lag==0)].flatten(), scale='log', stat='percent',  label='Wife surplus',color='red')
plt.legend()
#plt.savefig(root+'/Model/results/surplus_dist_i_2d.eps', format='eps', bbox_inches="tight")  
plt.show()


def mom(d,o=1):
    "Compute mean and higher order moments: mean (o=1), variance (o=2), skeweness (o=3), kurtosis (o=4)"

    if o==1:  return np.mean(d)
    elif o==2:return np.var(d)
    elif o==3:return skew(d)
    elif o==4:return kurtosis(d)
    else:     return "Error!!!" 

    

MΔC=np.array([mom(ΔC[sampl2].flatten(), o=i) for i in range(1,5)])
MΔCw=np.array([mom(ΔCw[sampl2].flatten(), o=i) for i in range(1,5)])
MΔCm=np.array([mom(ΔCm[sampl2].flatten(), o=i) for i in range(1,5)])
MΔws=np.array([mom(Δws[sampl2].flatten(), o=i) for i in range(1,5)])
MΔd=np.array([mom(Δd[sampl2].flatten(), o=i) for i in range(1,5)])
MΔYm=np.array([mom(ΔYm[sampl2].flatten(), o=i) for i in range(1,5)])
MΔYw=np.array([mom(ΔYw[sampl2].flatten(), o=i) for i in range(1,5)])


#Table with the moments
def p33(x): y=x;return str('%3.3f' % y)  
table=r'Wife, private consumption          & '+p33(MΔCw[0])+' & '+p33(MΔCw[1])+' & '+p33(MΔCw[2])+' & '+p33(MΔCw[3])+'    \\\\ '+\
      r'Husband, private consumption       & '+p33(MΔCm[0])+' & '+p33(MΔCm[1])+' & '+p33(MΔCm[2])+' & '+p33(MΔC[3])+'    \\\\ '+\
      r'Wife share of private consumption  & '+p33(MΔws[0])+' & '+p33(MΔws[1])+' & '+p33(MΔws[2])+' & '+p33(MΔws[3])+'    \\\\ '+\
      r'Home good expenditure              & '+p33(MΔd[0])+' & '+p33(MΔd[1])+' & '+p33(MΔd[2])+' & '+p33(MΔd[3])+'    \\\\ '+\
      r'Total consumption                  & '+p33(MΔC[0])+' & '+p33(MΔC[1])+' & '+p33(MΔC[2])+' & '+p33(MΔC[3])+'    \\\\ '+\
      r'Wife, earnings                     & '+p33(MΔYw[0])+' & '+p33(MΔYw[1])+' & '+p33(MΔYw[2])+' & '+p33(MΔYw[3])+'    \\\\ '+\
      r'Husband, earnings                  & '+p33(MΔYm[0])+' & '+p33(MΔYm[1])+' & '+p33(MΔYm[2])+' & '+p33(MΔYm[3])+'    \\\\\\bottomrule'
      
with open(root+'/Model/results/log_growth_moments.tex', 'w') as f: f.write(table); f.close() 



plt.plot(np.var(np.log(M.sim.incm[:,:par.Tr]),axis=0),label='Log men earnings')
plt.plot(np.var(np.log(M.sim.C_tot[:,:par.Tr]),axis=0),label='Log total consumption')
plt.plot(np.var(np.log(M.sim.dw[:,:par.Tr]),axis=0),label='Log hom good expenditures')
plt.legend()
plt.xlabel("Age")   
plt.ylim(0, 2.5)   
plt.savefig(root+'/Model/results/lifecycle_ineq1.eps', format='eps', bbox_inches="tight")                   
plt.show()

plt.plot(np.var(np.log(M.sim.incm[:,:par.Tr]),axis=0),label='Log men earnings')
plt.plot(np.var(np.log(M.sim.Cw[:,:par.Tr]),axis=0),label='Log w private consumption')
plt.plot(np.var(np.log(M.sim.Cm[:,:par.Tr]),axis=0),label='Log m private consumption')
plt.legend()
plt.xlabel("Age")   
plt.ylim(0, 2.5)   
plt.savefig(root+'/Model/results/lifecycle_ineq2.eps', format='eps', bbox_inches="tight")                   
plt.show()


########################
#Life-cycle behavior
######################

#Aggregate
keep=M.sim.couple<=1
plt.plot(np.mean(M.sim.dw,axis=0,where=keep),label="Public exp.")
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
plt.plot(np.mean(M.sim.dw,axis=0,where=keep),label="Public exp.")
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
plt.plot(np.mean(M.sim.dw,axis=0,where=keep),label="Public exp.")
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


list_a=[assets[M.sim.couple<=1],earnings[M.sim.couple<=1],M.sim.Cw[M.sim.couple==1],M.sim.Cm[M.sim.couple==1],M.sim.dw[M.sim.couple==1]]
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

    
sample = (age>age_initial[:,None]) & (age<=age_final[:,None]) & (M.sim.couple==1) & (M.sim.couple_lag==1)
