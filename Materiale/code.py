import scipy.stats as ss
from scipy.stats import chi2, norm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import math
import random
import statistics
from statistics import stdev

def uniformityTest(alpha=0.05):
    #creiamo 1000 valori random
    O = np.random.randint(0,10,100)
    #creiamo la tabella delle frequenze attese
    freq = (O==np.arange(10)[..., np.newaxis]).sum(axis=1)*1./O.size
    W, p_value = ss.chisquare(freq)
    print("W=",W,"p-value=",p_value)
    if(p_value > alpha):
        print("La distribuzione è uniforme con prob err", alpha*100,"%")
    else:
        print("La distribuzione non è uniforme con prob err", alpha*100,"%")
    return p_value

p_value=uniformityTest()

def displayChi(df, alpha, p, xP=16, yP=8):
    x=np.linspace(chi2.ppf(0, df),chi2.ppf(0.999, df), 100000)
    rv = chi2(df)
    region_limit = chi2.ppf(1-alpha, df)
    p_x = chi2.ppf(1-p, df)
    
    if(p_x > max(x)):
        p_min = max(x)-0.01
        p_max = max(x)
    else:
        p_min = p_x - 0.1
        p_max = p_x + 0.1
        
    NoCrit = x[x <= region_limit]
    Crit = x[x >= region_limit]
    AreaPValue = x[np.logical_and(x >= p_min, x <= p_max)]
    p_y = [0.01]*len(AreaPValue)
    
    plt.figure(figsize=(xP,yP))
    plt.plot(x, rv.pdf(x), 'k-', lw=1)
    plt.fill_between(NoCrit,chi2.pdf(NoCrit,df),color='g',alpha=0.5,linewidth=0,)
    plt.fill_between(Crit,chi2.pdf(Crit,df),color='r',alpha=0.8,linewidth=0,)
    plt.fill_between(AreaPValue,p_y,color='b',alpha=1,linewidth=0,)
    red_patch = mpatches.Patch(color='red', label='Regione Critica')
    blue_patch = mpatches.Patch(color='blue', label='p-value')
    green_patch = mpatches.Patch(color='green', label='Regione di accettazione')
    plt.legend(handles=[red_patch,blue_patch,green_patch], prop={'size': 20})
    
    plt.xlabel('t',  fontsize=20)
    plt.ylabel('f(t)',  fontsize=20)
    plt.xlim((0,max(x)))
    plt.show()

displayChi(df=99, alpha=0.05, p=p_value)


def updowntest(l, l_median, alpha=0.05):
    runs, n1, n2 = 0, 0, 0
    for i, item in enumerate(l):
        if( (l[i] >= l_median and l[i-1] < l_median) or (l[i] < l_median and l[i-1] >= l_median )):
            runs+=1
        if(l[i] >= l_median):
            n1+=1
        else:
            n2+=1
    
    run_exp = ((2*n1*n2)/(n1+n2))+1
    std_dev = math.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/(((n1+n2)**2)*(n1+n2-1)))
    
    z = (runs - run_exp)/std_dev
    z_alpha = ss.norm.ppf(1-alpha)
    return z, z-alpha, runs

def upAndDownGrafico(pvalue, xP=16, yP=8):
    sigma=stdev(l)
    U=(run-(2*n+1)/3)/sigma
    #Grafico
    plt.figure(figsize=(xP,yP))
    u = np.linspace(-3.05, +3.05, 100000)
    iq = norm(0,1)
    
    pvalue_pos = norm.ppf(1-pvalue)
    if pvalue_pos > 3:
        pvalue_minor = 3-0.005
        pvalue_major = 3
    elif pvalue_pos < -3:
        pvalue_minor = -3
        pvalue_major = -3+0.005
    else:
        pvalue_minor = pvalue_pos-0.005
        pvalue_major = pvalue_pos+0.005
    
 
    NoCrit = u[u <= norm.ppf(1-alpha)]
    coda1 = u[u >= norm.ppf(1-alpha)]
    AreaPValue = u[np.logical_and(u >= pvalue_minor,u <= pvalue_major)]
    
    pvalueY = [0.03]*len(AreaPValue)
    
    #Coloro le aree
    plt.plot(u, 1/(1 * np.sqrt(2 * np.pi)) *np.exp( - (u - 0)**2 / (2 * 1**2) ),linewidth=1, color='k')
    plt.fill_between(NoCrit,iq.pdf(NoCrit),color='g', alpha=0.5)
    plt.fill_between(coda1,iq.pdf(coda1),color='r')
    plt.fill_between(AreaPValue,pvalueY,color='b')
    
    #Creo la leggenda
    red_patch = mpatches.Patch(color='red', label='Regione critica')
    blue_patch = mpatches.Patch(color='blue', label='p-value')
    green_patch = mpatches.Patch(color='green', label='Regione \ndi accettazione')
    plt.legend(handles=[red_patch,blue_patch,green_patch], prop={'size': 20})
    
    #Mostro il grafico
    plt.xlabel('u',  fontsize=24)
    plt.ylabel('f(u)',  fontsize=24)
    plt.show()

#testiamo
l = []
alpha = 0.05
n=1000
for i in range(n):
    l.append(random.random())
l_median = statistics.median(l)
pvalue, p_alpha, run = updowntest(l, l_median, alpha)
pvalue = abs(pvalue)
p_alpha = abs(p_alpha)
print("p-value=",pvalue, "p_alpha=",p_alpha)
if(pvalue < p_alpha):
    print("Accetto l'ipotesi con prob err", alpha*100,"%")
else:
    print("Rifiuto l'ipotesi con prob err", alpha)
upAndDownGrafico(pvalue=pvalue)
