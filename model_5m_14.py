# -*- coding: utf-8 -*-
"""
Created on Sat Jul 01 10:05:39 2017

@author: LI-Cheng
"""

#加入齐纳击穿效应
import numpy as np
from matplotlib import pyplot as plt 
from IPython.core.pylabtools import figsize
import pymc as pm
from pymc.Matplot import plot as mcplot
import os
from scipy.stats.mstats import mquantiles
from separation_plot import separation_plot
from calculation_idc import planar_charge_0

#integrating 5m fluence data
np.set_printoptions(precision=2,suppress=False)
flu_08=[]
flu_20=[]
flu_40=[]
days=['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
months=['01','02','03','04','05','06','07','08','09','10','11','12']
years=['10','11']
for l,yy in enumerate(years):
    if l==0:months_1=months[4:12]
    else:months_1=months[0:8]
    for k,mm in enumerate(months_1):
        for j,dd in enumerate(days):
            if os.path.exists("20%s%s%s_Gp_part_5m.txt"% (yy,mm,dd))==True:
                data=np.genfromtxt("20%s%s%s_Gp_part_5m.txt"% (yy,mm,dd),skip_header=2,comments='#',usecols=(12,13,14))
                flu_5m_08=data[:,0]
                flu_5m_20=data[:,1]
                flu_5m_40=data[:,2]
                flu_08_1=[]
                flu_20_1=[]
                flu_40_1=[]
                for i in range(len(flu_5m_08)):
                    if flu_5m_08[i]>0:
                        flu_08_1.append(flu_5m_08[i])          
                for i in range(len(flu_5m_20)):
                    if flu_5m_20[i]>0:
                        flu_20_1.append(flu_5m_20[i])               
                for i in range(len(flu_5m_40)):
                    if flu_5m_40[i]>0:
                        flu_40_1.append(flu_5m_40[i])
                flu_08_1=np.array(flu_08_1)
                flu_20_1=np.array(flu_20_1)
                flu_40_1=np.array(flu_40_1)
                flu_08.append(flu_08_1.mean()*288*300)
                flu_20.append(flu_20_1.mean()*288*300)
                flu_40.append(flu_40_1.mean()*288*300)
            else:                
                break

flu_08=np.array(flu_08)
flu_20=np.array(flu_20)
flu_40=np.array(flu_40)    #Nn data
data_xxx=np.genfromtxt("xxx_1005_1108.csv",skip_header=1,usecols=(1,2,3,4),delimiter=",")
flu_06=data_xxx[:,0]
ano=data_xxx[:,2]
ano_avg=data_xxx[:,3]
'''
electric_field,voltage=planar_charge_0(0.6,2.0,flu_06,flu_20)

print electric_field,voltage
'''

'''
#FY-2C
data_fy=np.genfromtxt("FY_0411_0502.csv",skip_header=1,usecols=(1,2,3,4),delimiter=",")
flu_06=data_fy[:,0]
flu_20=data_fy[:,1]
ano=data_fy[:,2]
ano_avg=data_fy[:,3]
'''
flu_day=flu_20

#Accumulate fluence over days
day_accumulate=1
flu_acu=[]
for i in range(len(flu_day)):    
    if i<day_accumulate-1:
        flu_acu.append(flu_day[i]*day_accumulate)
        #flu_acu.append(flu_mdl[i])
    else:
        flu_acu.append(flu_day[i-day_accumulate+1:i+1].sum())
        #flu_acu.append(flu_mdl[i-day_acumulate+1:i+1].mean())
flu_acu=np.array(flu_acu)



flu_mdl=flu_acu

#Provide binomial data 

flu_bins=[]
for i in range(len(flu_mdl)):
    if ano_avg[i]==1:
        flu_bins.append(flu_mdl[i])
flu_bins=np.sort(flu_bins)
agr_bins=np.zeros(len(flu_bins))
D_bins=np.zeros(len(flu_bins))
for i in range(len(flu_bins)):
    for j in range(len(flu_mdl)):    
        if flu_mdl[j]>=flu_bins[i]:    
            agr_bins[i]+=1
            if ano_avg[j]==1:
                D_bins[i]+=1
nor_bins=agr_bins-D_bins
pr=D_bins.astype(float)/agr_bins.astype(float)


t=np.linspace(1,len(flu_mdl),len(flu_mdl))

#估计p_see的期望
ano_estimate=0
n_estimate=0
for i in range(len(flu_mdl)):
    if flu_mdl[i]<1e7:
        ano_estimate+=ano_avg[i]
        n_estimate+=1
estimated_p_see=ano_estimate/n_estimate

pi=3.141592654
eCharge=1.60217662E-19
lSpacing=3E-8
pCont=4.13566743E-15
eMass=9.10938215E-31
epsilon=2
def tunneling_rate(E):
	return eCharge*E*lSpacing*np.exp(-pow(pi,2)*eMass*lSpacing*pow(epsilon,2)/(pow(pCont,2)*eCharge*E))/pCont


#building model
iteration=150000
burn=iteration/2
thin=2
alpha=pm.Exponential('alpha',1.)
beta=pm.Exponential('beta',.1)
p_see=pm.Beta('p_see',1.,1.,value=estimated_p_see)

@pm.deterministic
def p_idc(flu_mdl=flu_mdl,alpha=alpha,beta=beta):
    return 1./(np.exp(-alpha*(np.log10(flu_mdl)-beta))+1)

@pm.deterministic
def p(flu_mdl=flu_mdl,p_see=p_see,p_idc=p_idc):
    return (p_see+p_idc-p_see*p_idc)

observation=pm.Bernoulli('obs',p,value=ano_avg,observed=True)
model=pm.Model([observation,alpha,beta,p_see])
_map=pm.MAP(model)
_map.fit()
mcmc=pm.MCMC(model)
mcmc.sample(iteration,burn,thin)
alpha_samples=mcmc.trace('alpha')[:,None]
beta_samples=mcmc.trace('beta')[:,None]
p_see_samples=mcmc.trace('p_see')[:,None]
alpha_mean=alpha_samples.mean()
beta_mean=beta_samples.mean()
p_see_mean=p_see_samples.mean()

mcplot(mcmc)
plt.show()

def prob(x,a,b,p):
    p_1=1./(np.exp(-a*(np.log10(x)-b))+1)
    p_2=p
    return (p_1+p_2-p_1*p_2)
def format_e(x):
    x_1=[]
    for i in range(len(x)):
        x_1.append('%.2e'%x[i])
    return x_1




figsize(12,5)
plt.title('accumulating %d day'%day_accumulate)
plt.subplot(311)
plt.bar(np.arange(len(flu_bins)),D_bins,0.5,color='red',label='Anomaly')
plt.bar(np.arange(len(flu_bins)),nor_bins,0.5,color='blue',bottom=D_bins,label='No anomaly')
plt.xticks(np.arange(len(flu_bins)),format_e(flu_bins),rotation='vertical')
plt.legend()
plt.subplot(312)
x=np.linspace(flu_mdl.min(),flu_mdl.max()*100,1e7)
y=prob(x,alpha_mean,beta_mean,p_see_mean)
plt.plot(x,y,color='red',label='average posterior')
plt.legend()
plt.scatter(flu_bins,D_bins.astype(float)/agr_bins.astype(float),s=30,c='k',alpha=0.7)
plt.xscale('log')
plt.subplot(313)
plt.scatter(flu_mdl,ano_avg,s=30,c='k',alpha=0.7)
plt.plot(x,y,color='red',label='average posterior')
plt.legend()
plt.xscale('log')
plt.show()


#simulated datasets
simulated=pm.Bernoulli('bernoulli_sim',p)
mcmc=pm.MCMC([simulated,alpha,beta,p_see,observation])
mcmc.sample(100000,70000)
simulations=mcmc.trace('bernoulli_sim')[:].astype(int)
print simulations.shape

'''
figsize(12,5)
plt.title('Simulated datasets using posterior parameters')
for i in range(4):
    ax=plt.subplot(4,1,i+1)
    plt.scatter(t,simulations[10000*i,:],color='k',s=50,alpha=0.3)
plt.show()
'''

#用后验样本产生二项数据
figsize(12.5,6)
ano_sim=np.zeros(len(flu_mdl))
agr_sim=np.zeros(len(flu_mdl))
for i in range(simulations.shape[0]):
    for j in range(len(flu_mdl)):
        for k in range(len(flu_mdl)):
            if flu_mdl[k]>=flu_mdl[j]:
                agr_sim[j]+=1
                if simulations[i,k]==1:
                    ano_sim[j]+=1 
plt.scatter(flu_mdl,ano_sim/agr_sim,s=30,alpha=0.5)
plt.xscale('log')
plt.show()

    
'''
#separation plot
figsize(10,1.5)
posterior_probability=simulations.mean(axis=0)
separation_plot(posterior_probability,ano_avg)
plt.title('Separation plot of anomaly,expected numbers of anomaly:%f'%simulations.mean(axis=0).sum())
plt.show()

#expect and variance
expect_samples=sum(ano_avg)/len(ano_avg)
expect_model=0.
for i in range(simulations.shape[0]):
    expect_model+=sum(simulations[i,:])
expect_model=expect_model/(simulations.shape[0]*simulations.shape[1])
variance_samples=0.
for i in range(len(ano_avg)):
    variance_samples+=(ano_avg[i]-expect_samples)**2
variance_samples=variance_samples/len(ano_avg)
variance_model=0.
for i in range(simulations.shape[0]):
    for j in range(simulations.shape[1]):
        variance_model+=(simulations[i,j]-expect_model)**2
variance_model=variance_model/(simulations.shape[0]*simulations.shape[1])

#Brier Score
bs_flu=0
for i in range(simulations.shape[1]):
        bs_flu+=(simulations.mean(axis=0)[i]-ano_avg[i])**2
bs_flu=bs_flu/simulations.shape[1]

#95 CI
figsize(12,5)
plt.subplot(211)
plt.scatter(flu_mdl,ano_avg,s=30,c='k',alpha=0.7,label='Expect of samples: %f\nVariance of samples:%f'%(expect_samples,variance_samples))
plt.legend()
plt.xscale('log')
plt.subplot(212)
x=np.linspace(flu_mdl.min(),flu_mdl.max()*100,1e7)
y=prob(x,alpha_mean,beta_mean,p_see_mean)
x_0=np.linspace(flu_mdl.min(),flu_mdl.max()*10,1e3)[:,None]
p_qs=prob(x_0.T,alpha_samples,beta_samples,p_see_mean) 
qs=mquantiles(p_qs,[0.025,0.975],axis=0)
plt.fill_between(x_0[:,0],*qs,color='#7A68A6',alpha=0.7)
plt.plot(x_0[:,0],qs[0],color='#7A68A6',alpha=0.7,label='Expect of model: %f\nVariance of model:%f\nBrier Score:%f'%(expect_model,variance_model,bs_flu))
plt.plot(x,y,lw=1,ls='--',color='k',label='average posterior')
plt.scatter(flu_mdl,ano_avg,s=20,c='k',alpha=0.5)
plt.title('Flux Model')
plt.legend()
plt.xscale('log')
plt.show()

figsize(12,5)
x_1=np.linspace(flu_mdl.min(),flu_mdl.max()*10,1e3)[:,None]
qs_1=mquantiles(p_see_samples*np.ones(1e3).T,[0.025,0.975],axis=0)
plt.fill_between(x_1[:,0],*qs_1,color='#7A68A6',alpha=0.7)
plt.plot(x_1[:,0],qs_1[0],label='95% CI',color='#7A68A6',alpha=0.7)
plt.plot(x,p_see_mean*np.ones(1e7),lw=1,ls='--',color='k',label='average posterior')
plt.title('Estimated p_see:%f and Mean of posterior:%f'%(estimated_p_see,p_see_mean))
plt.legend()
plt.xscale('log')
plt.ylim(0,1)
plt.show()




print 'Acumulated days:',day_accumulate
print 'Estimated p_see:',estimated_p_see 
#print 'Mean probability of anomaly over >fluence:',pr.mean()
print 'Posterior of parameters:',alpha_mean,beta_mean,p_see_mean
print 'Brier Score of flu_model:',bs_flu
print 'Expected numbers of anomaly:',simulations.mean(axis=0).sum()
print 'Expect and variance of samples:',expect_samples,variance_samples
print 'Expect and variance of model:',expect_model,variance_model
'''
