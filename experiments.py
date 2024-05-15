import numpy as np
import pickle as pkl
import pandas as pd

from sklearn.tree import DecisionTreeRegressor
from pysyncon import Dataprep, Synth
import cvxpy as cp

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as mtrans

## specify plot options 
plt.rcParams.update({
    'axes.linewidth' : .5,
    'font.size': 9.9, 
    "text.usetex": True,
    'font.family' : 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.weight': 'heavy',
    'text.latex.preamble' : r'\usepackage{amsmath,amsfonts}'})

## custom color palette
lblue = (40/255,103/255,178/255)
cred  = (177/255, 4/255, 14/255)

from collections import defaultdict
results  = defaultdict(list)
betas = {}

# subsampling rate, common to all experiments
p = 0.5

# number of bags, common to all experiments
# 10,000 is BIG, experiment 3 is esp. slow
B = 10000

## subsample m out of [n] without replacement B times
def subsample(n, m, B):
    idx = np.zeros((B, m), dtype=int)
    for b in range(B):
        row = np.random.choice(n, m, replace=False)
        idx[b, :] = row
    return(idx)

#####################################
### Experiment 1: decision trees ####
#####################################

## base algorithm for regression trees
## where max_depth=50
def fit_tree_(Z_tr):
    n = Z_tr.shape[0]
    X_tr, y_tr = Z_tr[:, :-1], Z_tr[:, -1]

    m = DecisionTreeRegressor(max_depth=50)
    m.fit(X_tr, y_tr)
    return(m)

np.random.seed(1234567891)
n, d = 500, 40
X = np.random.rand(n, d)**2
y = np.zeros(n)
for j in range(d):
    y += np.sin(X[:, j]/(1+j))
y[::4] += 2*(.5-np.random.rand(int(n/4)))
y[::3] += .5*(.5-np.random.rand(1+n//3))
y = (y-y.min())/(y.max()-y.min())

n_te = 1000
x_te = np.random.rand(n_te, d)

Z = np.hstack([X, y[:, np.newaxis]])

nm = 'DT, n=%d' % n
print(nm)
m_full = fit_tree_(Z)
yhat = m_full.predict(x_te)

loo, rs = [], []
for i in range(n):
    idx = np.delete(np.arange(n), i)
    m_noti = fit_tree_(Z[idx])
    y_noti = m_noti.predict(x_te)
    loo.append(np.mean(np.abs(yhat - y_noti)**2)**.5)
results[nm] = np.sort(loo)[::-1]

m = int(n*p)
idx  = subsample(n, m, B)
yhat = np.zeros((B, n_te))
for b in range(B):
    mb = fit_tree_(Z[idx[b], :])
    yhat[b] = mb.predict(x_te)
    
nm = 'Subbagged DT, n=%d' % n
print(nm)
y_full = np.mean(yhat, 0)
loo = []
for i in range(n):
    # average over bags not containing i
    y_noti = np.mean(yhat[~np.any(idx==i, axis=1)], 0)
    loo.append(np.mean(np.abs(y_full - y_noti)**2)**.5) 
results[nm] = np.sort(loo)[::-1]
betas[nm] = 1/np.sqrt(4*n)

#####################################
######## Experiment 2: SCM ##########
#####################################

df = pd.read_csv("germany.csv")
controls = [
    "USA",
    "UK",
    "Austria",
    "Belgium",
    "Denmark",
    "France",
    "Italy",
    "Netherlands",
    "Norway",
    "Switzerland",
    "Japan",
    "Greece",
    "Portugal",
    "Spain",
    "Australia",
    "New Zealand",
]

NC = len(controls)
res = {}

for j in range(NC+1):
    print(j)
    controls_ = controls.copy()
    df_ = df.copy()
    if j<NC:
        controls_.remove(controls[j])
        df_ = df_.loc[df_['country'] != controls[j]]
    
    dataprep_train = Dataprep(
        foo=df_,
        predictors=["gdp", "trade", "infrate"],
        predictors_op="mean",
        time_predictors_prior=range(1971, 1981),
        special_predictors=[
            ("industry", range(1971, 1981), "mean"),
            ("schooling", [1970, 1975], "mean"),
            ("invest70", [1980], "mean"),
        ],
        dependent="gdp",
        unit_variable="country",
        time_variable="year",
        treatment_identifier="West Germany",
        controls_identifier=controls_,
        time_optimize_ssr=range(1981, 1991),
    )

    synth_train = Synth()
    synth_train.fit(dataprep=dataprep_train)
    
    dataprep = Dataprep(
        foo=df_,
        predictors=["gdp", "trade", "infrate"],
        predictors_op="mean",
        time_predictors_prior=range(1981, 1991),
        special_predictors=[
            ("industry", range(1981, 1991), "mean"),
            ("schooling", [1980, 1985], "mean"),
            ("invest80", [1980], "mean"),
        ],
        dependent="gdp",
        unit_variable="country",
        time_variable="year",
        treatment_identifier="West Germany",
        controls_identifier=controls_,
        time_optimize_ssr=range(1960, 1990),
    )
    
    synth = Synth()
    synth.fit(dataprep=dataprep, custom_V=synth_train.V)

    name = 'full' if j==NC else controls[j]
    res[name] = synth.weights()

loo = []
for j in range(NC):
    name = controls[j]
    
    w_i = pd.concat([res[name], pd.Series({name:0})])
    loo.append(np.linalg.norm(w_i - res['full']))
nm = 'SCM, n=16'
print(nm)
results[nm] = np.sort(loo)[::-1]
    
m = int(p*n)
bags = []
weights = []
for b in range(B):
    r = np.random.choice(NC, size=(m,), replace=False)
    bags.append(r)
    Dr = pd.concat([df.loc[df['country'] == controls[j]] for j in r] + [df.loc[df['country'] == 'West Germany']])
    controls_ = [controls[j] for j in set(r)]
    
    dataprep_train = Dataprep(
        foo=Dr,
        predictors=["gdp", "trade", "infrate"],
        predictors_op="mean",
        time_predictors_prior=range(1971, 1981),
        special_predictors=[
            ("industry", range(1971, 1981), "mean"),
            ("schooling", [1970, 1975], "mean"),
            ("invest70", [1980], "mean"),
        ],
        dependent="gdp",
        unit_variable="country",
        time_variable="year",
        treatment_identifier="West Germany",
        controls_identifier=controls_,
        time_optimize_ssr=range(1981, 1991),
    )

    synth_train = Synth()
    synth_train.fit(dataprep=dataprep_train)
    
    dataprep = Dataprep(
        foo=Dr,
        predictors=["gdp", "trade", "infrate"],
        predictors_op="mean",
        time_predictors_prior=range(1981, 1991),
        special_predictors=[
            ("industry", range(1981, 1991), "mean"),
            ("schooling", [1980, 1985], "mean"),
            ("invest80", [1980], "mean"),
        ],
        dependent="gdp",
        unit_variable="country",
        time_variable="year",
        treatment_identifier="West Germany",
        controls_identifier=controls_,
        time_optimize_ssr=range(1960, 1990),
    )
    
    synth = Synth()
    synth.fit(dataprep=dataprep, custom_V=synth_train.V)
    weights.append(synth.weights())
    
weights_filled = []
for b in range(B):
    r = bags[b]
    w = weights[b]
    rc = [j for j in range(NC) if j not in r]
    weights_filled.append(pd.concat([w, pd.Series({controls[j]:0 for j in rc})]))
    
from functools import reduce

w_full = reduce(lambda x, y: x.add(y, fill_value=0), weights_filled)/B

loo = []
for j in range(NC):
    bs_j = [b for (b, r) in enumerate(bags) if j not in r]
    B_j = len(bs_j)
    w_j = reduce(lambda x, y: x.add(y, fill_value=0), [weights_filled[b] for b in bs_j])/B_j
    loo.append(np.linalg.norm(w_j-w_full))
    
nm = 'Subbagged SCM, n=16'
print(nm)
results[nm] = np.sort(loo)[::-1]
betas[nm] = 1/np.sqrt(n)

#####################################
######## Experiment 3: LSSA #########
#####################################

n = 200

np.random.seed(1234567891)
x = ((2*np.random.rand(n)-1)*(2*np.random.rand(n)-1))
Z = np.random.randn(n)
f_= np.sin(2/(1+x)+2/(1-x))
y = f_ + .1*Z

D = int(n/2)
Nk = np.arange(-D, D+1)
K = len(Nk)

m = int(n*p)
p = m/n
D_ = int(m/2)
Nk_ = np.arange(-D_, D_+1)
K_ = len(Nk_)

# Psi matrix for base algorithm
PSI = np.zeros((n,K))
for i in range(n):
    for k in range(K):
        PSI[i,k] = np.exp(1j*Nk[k]*x[i])
        
# Psi matrix for subsampled algorithm
PSI_ = np.zeros((n,K_))
for i in range(n):
    for k in range(K_):
        PSI_[i,k] = np.exp(1j*Nk_[k]*x[i])
        
s = 2
D = (1+Nk**2)**s
D_= (1+Nk_**2)**s

def A(PSI, y, R=10):
    n, K = PSI.shape
    b = cp.Variable(K)
    cost = cp.sum_squares(y - PSI@b)
    prob = cp.Problem(cp.Minimize(cost), 
                      ([cp.sum_squares(b)<=R**2]))
    prob.solve()

    return(b.value)

# run base algo
b = A(PSI@np.diag(np.sqrt(D)), y)

## compute L1 leave-one-out perturbations
## for the base algorithm
loo = []
for i in range(n):
    ic = np.delete(np.arange(n), i)
    b_ = A(PSI[ic]@np.diag(np.sqrt(D)), y[ic])
    loo.append(np.linalg.norm(b-b_))
nm = 'LSSA, n=200'
print(nm)
results[nm] = np.sort(loo)[::-1]
    
## compute L1 leave-one-out perturbations
## for the subbagged algorithm
B = 10000
m = int(n*p)
p = m/n

bags = np.zeros((B, m)).astype(int)
w_bags = []
for b in range(B):
    bag = np.random.choice(n, m, False)
    bags[b] = bag
    w_bags.append(A(PSI_[bag]@np.diag(np.sqrt(D_)), y[bag]))
w_bags = np.array(w_bags)

w_subb_full = np.mean(w_bags, axis=0)
w_subb_loo  = []
loo   = []
for i in range(n):
    ic = np.delete(np.arange(n), i)
    w_i = np.mean(w_bags[np.all(bags!=i,1)], axis=0)
    w_subb_loo.append(w_i)
    loo.append(np.linalg.norm(w_i - w_subb_full))
nm = 'Subbagged LSSA, n=200'
print(nm)
results[nm] = np.sort(loo)[::-1]
betas[nm] = 10/np.sqrt(n-1)

#####################################
####### Experiment 4: softmax #######
#####################################

# define base algo
def A(X):
    n, d = X.shape
    x = np.sum(X, 0)
    w = np.exp(x)/np.sum(np.exp(x))
    return(w)

n = 2000
d = 100
m = int(p*n)

# sample X_{ij} ~ iid Bern(.2)
np.random.seed(1234567891)
X = (np.random.rand(n, d) < .2)*1

## compute L1 leave-one-out perturbations
## for the base algorithm
w_base_full = A(X)
w_base_loo  = []
loo   = []
for i in range(n):
    ic = np.delete(np.arange(n), i)
    w_i = A(X[ic])
    w_base_loo.append(w_i)
    loo.append(np.linalg.norm(w_i-w_base_full, 1)/2)
nm = 'L1, n=%d' % n
print(nm)
results[nm] = np.sort(loo)[::-1]


bags = np.zeros((B, m)).astype(int)
w_bags = []
for b in range(B):
    bag = np.random.choice(n, m, False)
    bags[b] = bag
    w_bags.append(A(X[bag]))
w_bags = np.array(w_bags)

w_subb_full = np.mean(w_bags, axis=0)
w_subb_loo  = []
loo   = []
for i in range(n):
    ic = np.delete(np.arange(n), i)
    w_i = np.mean(w_bags[np.all(bags!=i,1)], axis=0)
    w_subb_loo.append(w_i)
    loo.append(np.linalg.norm(w_i - w_subb_full, 1)/2)

nm = 'Subbagged LSSA, n=%d' % n
print(nm)
results[nm] = np.sort(loo)[::-1]

Hn = np.sum([1/i for i in range(1, n+1)])
eta = (Hn-1)/(n-Hn)

a = np.sqrt((2/n)*(1+eta)*(p/(1-p)+4*eta/(1-p)**2)*(np.log(2*d)) )
C = (p/(1-p))*(2/3)*(1+eta)*np.log(2*d)/n
beta = (a+np.sqrt(a**2+4*C))/2
betas[nm] = beta

#####################################
### Mean-square stability results ###
#####################################

for k in results:
    print(k, np.mean(np.array(results[k])**2))

#####################################
###### Tail stability results #######
#####################################

keys = [('DT, n=500', 'Subbagged DT, n=500'), 
         ('SCM, n=16', 'Subbagged SCM, n=16'), 
         ('LSSA, n=200', 'Subbagged LSSA, n=200'), 
         ('L1, n=2000', 'Subbagged L1, n=2000')]
ns = [500, 16, 200, 2000]
xlabs = [
    'Leave-one-out perturbation $\\|\\hat{w} - \\hat{w}^{\\setminus i}\\|_{L_2}$',
    'Leave-one-out perturbation $\\|\\hat{w} - \\hat{w}^{\\setminus i}\\|_{2}$',
    'Leave-one-out perturbation $\\|\\hat{w} - \\hat{w}^{\\setminus i}\\|_{2,2}$',
    'Leave-one-out perturbation $\\text{d}_{\\text{TV}}(\\hat{w}, \\hat{w}^{\\setminus i})$',
]

NBINS = 80
bins = [np.linspace(0, .5, NBINS), np.linspace(0, 1, NBINS), np.linspace(0, 10, NBINS), np.linspace(0, .5, NBINS)]

xlims = [[0, .15], [0, .6], [0, 15], [0, .25]]

LINEWIDTH=1

BASE_ALGO_LEFT  = 'Base algorithm'
SUBB_ALGO_LEFT = 'Subbagged algorithm'

### make small plots
TEXTWIDTH = 6.29651*.95
for PLOT_NUM, (base_key, subb_key) in enumerate(keys):
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, TEXTWIDTH*8/(7*4)), frameon=False, sharex=True)

    n = ns[PLOT_NUM]
    axs[0].hist(results[base_key], 
                  bins=bins[PLOT_NUM], color=cred, alpha=.6, 
                  label=BASE_ALGO_LEFT)  
    axs[0].hist(results[subb_key], 
                  bins=bins[PLOT_NUM], color=lblue, alpha=.6, 
                  label=SUBB_ALGO_LEFT)

    axs[0].set_xlim([0, .15])
    axs[1].step(results[base_key], 
                  np.arange(n)/n, lw=LINEWIDTH, c=cred, 
                  label='Base algorithm', where='pre')
    k = 10000
    delta = np.arange(k)/k
    axs[1].step(results[subb_key], 
                  np.arange(n)/n, lw=LINEWIDTH, c=lblue, 
                  label='Subbagging', where='pre')
    axs[1].plot(betas[subb_key]/np.sqrt(delta), delta, 'k', ls='dotted', lw=LINEWIDTH, 
                  label='Stability guarantee\nfor subbagging')

    axs[1].set_ylim([0, 1.01])
    axs[1].set_xlim([0.0, .15])

    fig.tight_layout()
    plt.subplots_adjust(hspace = 1)


    axs[1].set_xlabel('$\\varepsilon$') # Error tolerance 
    if PLOT_NUM==0:
        axs[1].legend(fontsize=7, loc='upper right')
        axs[0].legend(fontsize=7, loc='upper right')
    axs[1].set_ylabel('$\\delta$') # Error probability 
    axs[0].set_ylabel('Frequency')

    axs[0].set_xlabel(xlabs[PLOT_NUM])
    axs[0].set_xlim(xlims[PLOT_NUM])

    axs[1].set_ylim([0, .501])

    plt.savefig('fig1' + 'abcd'[PLOT_NUM] + '.pdf', bbox_inches='tight')

    plt.show()