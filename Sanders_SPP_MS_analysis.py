#####################################
### Initial imports
#####################################

import pandas as pd
import numpy as np
import pickle, os, copy
import scipy
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FixedLocator, MaxNLocator, AutoMinorLocator
import statsmodels.api as sm

plt.ion()

## Run sampler, rather than loading previously exported samples?
sample_on = 0

## Maximum year of dataset
max_year = 2017

## See https://github.com/akucukelbir/stanhelper - using version 0.8 for python2 compatibility
import stanhelper
import subprocess
# Point this path to a local cmdstan installation
cmdstan_path = os.path.expanduser('~/cmdstan-2.17.1/')

from scipy import stats as sstats

## Location to write out key statistics presented in the manuscript in tex format
fact_file = 'SPP_MS_facts.tex'


#####################################
### Load and prepare data
#####################################

## Load data
## Obtained from https://www.motherjones.com/politics/2012/12/mass-shootings-mother-jones-full-data/
data = pd.read_excel('MotherJonesData_2018_02_24.xlsx','US mass shootings')

## Stadardize on definition of fatalities at 4.  Mother Jones changed it to 3 in 2013.
data = data[data.Fatalities > 3]

## Prepare data
# Aggregate data anually
data_annual = data.groupby('Year')
# Count cases by year and fill in empty years
cases_resamp = data_annual.count().Case.ix[np.arange(1982,max_year+1)].fillna(0)
# Enumerate years in range
data_years = cases_resamp.index.values
# Enumerate quarters across daterange for later plotting
data_years_samp = np.arange(min(data_years), max(data_years)+10, .25)
# Format for Stan
stan_data = {
	'N1': len(cases_resamp),
	'x1': data_years - min(data_years),
	'z1': cases_resamp.values.astype(int),
	'N2': len(data_years_samp),
	'x2': data_years_samp - min(data_years),
	}


#####################################
### Compile stan model
#####################################

### Compile using cmdstan
if sample_on:
	### Script expects cmdstan installation at cmdstan_path
	subprocess.call("mkdir "+cmdstan_path+"user-models", shell=1)
	subprocess.call("cp gp_model_final.stan " + cmdstan_path+"user-models/", shell=1)
	subprocess.call("make user-models/gp_model_final", cwd=cmdstan_path, shell=1)	

## Sampling parameters
Nchains = 20
Niter = 8000
cdic = {'max_treedepth': 15, 'adapt_delta': 0.95}



#####################################
### Setup inputs
#####################################

## Sample with strong prior on rho
stan_data_rho_strong = copy.copy(stan_data)
stan_data_rho_strong['alpha_rho'] = 4
stan_data_rho_strong['beta_rho'] = 1

stan_data_rho_weak = copy.copy(stan_data)
stan_data_rho_weak['alpha_rho'] = 1
stan_data_rho_weak['beta_rho'] = 1/100.



#####################################
### Run model
#####################################

if sample_on:

	### Sample with cmdstan
	## Delete any old samples first
	os.system('rm output_cmdstan_gp_rhostrong_samples*.csv')
	stanhelper.stan_rdump(stan_data_rho_strong, 'input_data_rhostrong_final.R')
	p = []
	for i in range(Nchains):
		cmd = """
	{0}user-models/gp_model_final \
	data file='input_data_rhostrong_final.R' \
	sample num_warmup={2} num_samples={2} \
	adapt delta={4} \
	algorithm=hmc engine=nuts max_depth={3} \
	random seed=1002 id={1} \
	output file=output_cmdstan_gp_rhostrong_samples{1}.csv
				""".format(cmdstan_path, i+1, Niter/2, cdic['max_treedepth'], cdic['adapt_delta'])
		p += [subprocess.Popen(cmd, shell=True)]

	## Don't move on until sampling is complete.
	for i in range(Nchains):
		p[i].wait()


	## Sample with weak prior on rho
	
	## Sample with cmdstan
	## Delete any old samples first
	os.system('rm output_cmdstan_gp_rhoweak_samples*.csv')
	stanhelper.stan_rdump(stan_data_rho_weak, 'input_data_rhoweak_final.R')
	p = []
	for i in range(Nchains):
		cmd = """
	{0}user-models/gp_model_final \
	data file='input_data_rhoweak_final.R' \
	sample num_warmup={2} num_samples={2} \
	adapt delta={4} \
	algorithm=hmc engine=nuts max_depth={3} \
	random seed=1002 id={1} \
	output file=output_cmdstan_gp_rhoweak_samples{1}.csv
				""".format(cmdstan_path, i+1, Niter/2, cdic['max_treedepth'], cdic['adapt_delta'])
		p += [subprocess.Popen(cmd, shell=True)]

	## Don't move on until sampling is complete.
	for i in range(Nchains):
		p[i].wait()



def stan_read_csv_multi(path):
	"""
	Wrap the stanhelper.stan_read_csv function to load outputs
	from multiple chains.
	
	Parameters:
	* path: file path for cmdstan output files including wildcard (*)
	"""
	## Enumerate files
	from glob import glob
	files = glob(path)
	
	## Read in each file
	result = {}
	for file in files:
		result[file] = stanhelper.stan_read_csv(file)
	
	## Combine dictionaries
	result_out = {}
	keys = result[files[0]]
	for key in keys:
		result_out[key] = result[files[0]][key]
		for f in files:
				result_out[key] = np.append(result_out[key], result[f][key], axis=0)

	## Remove extraneous dimension
	for key in keys:
		if result_out[key].shape[-1] == 1:
			result_out[key] = np.squeeze(result_out[key], -1)
	
	return result_out

stan_model_ext_rho_strong = stan_read_csv_multi('output_cmdstan_gp_rhostrong_samples*.csv')
stan_model_ext_rho_weak = stan_read_csv_multi('output_cmdstan_gp_rhoweak_samples*.csv')


#####################################
### Load summary of fit
#####################################

def read_stansummary(path, cmdstan_path=cmdstan_path):
	"""
	Wrapper for the cmdstan program stan_summary to calculate
	sampling summary statistics across multiple MCMC chains.
	
	Args:
		path (str): Path, with a wildcard (*) for the id number
		of each output chain
		
		cmdstan_path (str): Path to the stan home directory
	
	Returns:
		out: A pandas dataframe with the summary statistics provided 
		by stan_summary.  Note that each element of array variables
		are provided on separate lines
	"""
	from StringIO import StringIO
	summary_string = subprocess.check_output(cmdstan_path + 'bin/stansummary --sig_figs=5 '+path, shell=1)
	out = pd.read_table(StringIO(summary_string), sep='\s+', header=4, skip_footer=6, engine='python')
	return out

## Use cmdstan's stansummary command to calculate rhat
stan_model_sum_rho_strong = read_stansummary('output_cmdstan_gp_rhostrong*.csv')
stan_model_sum_rho_weak = read_stansummary('output_cmdstan_gp_rhoweak*.csv')

## Get summary statistics using cmdstan wrapper
model_summary = stan_model_sum_rho_strong
Rhat_vec = stan_model_sum_rho_strong['R_hat'].values
Rhat_vec_weak = stan_model_sum_rho_weak['R_hat'].values
pars = stan_model_sum_rho_strong.index

## Replace y1, y2 with summaries
sel_pars = ['y1', 'y2', u'eta_sq', u'inv_rho', u'sigma_sq', u'mu_0', u'mu_b', 'NB_phi_inv']
Rhat_tex_names = {
	'y1':'$y_1$',
	'y2':'$y_2$',
	u'eta_sq':'$\\eta^2$',
	u'inv_rho':'$\\rho^{-1}$',
	u'sigma_sq':'$\\sigma^2$',
	u'mu_0':'$\\mu_0$',
	u'mu_b':'$\\mu_b$',
	'NB_phi_inv':'$\\rm{NB}_{\\phi^{-1}}$',
	}
Rhat_dic = {}
Rhat_dic_weak = {}
for spar in sel_pars:
	if spar in ('y1','y2'):
		sel = np.where([True if p.startswith(spar) else False for p in pars])
		Rhat_dic[spar] = np.percentile(Rhat_vec[sel], [5,50,95])
		Rhat_dic_weak[spar] = np.percentile(Rhat_vec_weak[sel], [5,50,95])
	else:
		Rhat_dic[spar] = [Rhat_vec[[pars==spar]],]*3
		Rhat_dic_weak[spar] = [Rhat_vec_weak[[pars==spar]],]*3



plt.figure(figsize=(5,6))
plt.errorbar(np.array(Rhat_dic.values())[:,1], np.arange(len(sel_pars)), \
	xerr= [np.array(Rhat_dic.values())[:,1] - np.array(Rhat_dic.values())[:,0],\
		np.array(Rhat_dic.values())[:,2] - np.array(Rhat_dic.values())[:,1]],\
	capsize=0, marker='o', color='k', lw=0, label='Strong prior on $\\rho$')
plt.errorbar(np.array(Rhat_dic_weak.values())[:,1], np.arange(len(sel_pars)), \
	xerr= [np.array(Rhat_dic_weak.values())[:,1] - np.array(Rhat_dic_weak.values())[:,0],\
		np.array(Rhat_dic_weak.values())[:,2] - np.array(Rhat_dic_weak.values())[:,1]],\
	capsize=0, marker='o', mfc='none', mec='.5', lw=0, label='Weak prior on $\\rho$')
plt.yticks(np.arange(len(sel_pars)), [Rhat_tex_names[k] for k in Rhat_dic.keys()], size=11)
plt.xlabel('$\\hat{R}$')
plt.axvline(1.0, color='.5', ls='solid', zorder=-2)
plt.axvline(1.05, color='.5', ls='dashed', zorder=-2)
plt.ylim(-.5, len(sel_pars)-.5)
plt.xlim(0.99, 1.06)
plt.legend()

plt.savefig('fig_rhat.pdf', bbox_inches='tight')



#####################################
### Diagnostic plots
#####################################

## Traceplot
trace_pars = [('eta_sq','$\\eta^2$'),
		  ('inv_rho','$\\rho^{-1}$'),
		  ('sigma_sq','$\\sigma^2$'),
		  ('mu_0','$\\mu_0$'),
		  ('mu_b','$\\mu_b$'),
		  ('NB_phi_inv','$\\rm{NB}_{\\phi^{-1}}$')]
fig,axs = plt.subplots(len(trace_pars),2, figsize=(8,8), sharex='all', sharey='row')
exts = [stan_model_ext_rho_strong, stan_model_ext_rho_weak]
exts_names = [r'Strong $\rho$ prior', r'Weak $\rho$ prior']
for j in range(2):
	axs[0,j].set_title(exts_names[j])
	for i,par in enumerate(trace_pars):
		axs[i,j].plot(exts[j][par[0]], color='.5')
		if j==0: axs[i,j].set_ylabel(par[1])
		for k in range(1, Nchains+1):
			axs[i,j].axvline(Niter/2 * k, c='b', zorder=-1, alpha=0.5)

	axs[len(trace_pars) - 1,j].set_xticks(np.arange(0, (Niter/2)*Nchains+1, Niter*2))

plt.savefig('fig_traceplot.pdf', bbox_inches='tight')





N_samp = Niter / 2

fig, axs = plt.subplots(5,5, figsize=(7,7), sharex='all', sharey='all')
po = axs[0,0].plot(data_years, stan_data['z1'], 'o', c='k', mfc='k', label='Observations', zorder=2, lw=1, ms=4)
axs[0,0].legend(numpoints=1, prop={'size':6})
for i in range(1,25):
	draw = np.random.randint(0, N_samp)
	py = stan_model_ext_rho_strong['z_rep'][draw][:stan_data['N1']]
	axs.flatten()[i].plot(data_years, py,  mfc='k', marker='o',
	  lw=.5, mec='none', ms=2, color='.5', label='GP realization')
axs[0,1].legend(numpoints=1, prop={'size':6})
axs[0,0].set_ylim(0,15)
axs[0,0].set_xticks([1980, 1990, 2000, 2010, 2020])
for ax in axs.flatten(): 
	plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=9)
	plt.setp(ax.get_yticklabels(), fontsize=9)

axs[2,0].set_ylabel('Public mass shootings per year', size=9)
axs[0,0].set_title(r'Strong prior on $\rho^{-1}$')

plt.savefig('fig_ppc.pdf', bbox_inches='tight')



#####################################
### Visualize GP
#####################################


def plot_GP(stan_model_ext):
	y2_sum = np.percentile(np.exp(stan_model_ext['y2']), [16,50,84], axis=0)
	plt.figure(figsize=(7,5))
	pfb = plt.fill_between(data_years_samp, y2_sum[0], y2_sum[2], color='b', alpha=.5)
	pfg = plt.plot(data_years_samp, y2_sum[1], c='b', lw=2, label='GP model', zorder=0)
	po = plt.plot(data_years, stan_data['z1'], 'o', c='k', label='Observations', zorder=2)
	plt.xlabel('Year')
	plt.ylabel('Annual rate of public mass shootings')
	plt.legend(prop={'size':10}, loc=2)
	plt.ylim(0,15)
	plt.gca().xaxis.set_minor_locator(FixedLocator(np.arange(min(data_years_samp), max(data_years_samp))))
	plt.gca().set_xlim(min(data_years_samp) - 1, max(data_years_samp) + 1)
	return pfb, pfg, po

pfb, pfg, po = plot_GP(stan_model_ext_rho_strong)
plt.title(r'Strong prior on $\rho^{-1}$')
plt.savefig('fig_GP_strong.pdf', bbox_inches='tight')



def plot_GP_mu_draws(stan_model_ext):
	plot_GP(stan_model_ext)
	N_samp = len(stan_model_ext['mu_0'])
	px = np.linspace(min(data_years_samp), max(data_years_samp), 100) 
	pfms = []
	for i in range(20):
		draw = np.random.randint(0, N_samp)
		py = np.exp(stan_model_ext['mu_0'][draw] + (px - min(data_years)) * stan_model_ext['mu_b'][draw])
		pfms.append(plt.plot(px, py,  c='r', alpha=0.5,
		  zorder = 1, label = 'Mean function draws' if i==0 else None))
	plt.legend(prop={'size':10}, loc=2)

plot_GP_mu_draws(stan_model_ext_rho_strong)
plt.title(r'Strong prior on $\rho^{-1}$')
plt.savefig('fig_GP_strong_draws.pdf', bbox_inches='tight')





y2_gp_rho_strong = np.percentile(np.exp(
		stan_model_ext_rho_strong['y2'] - 
		np.dot(stan_model_ext_rho_strong['mu_b'][:,np.newaxis], (data_years_samp[np.newaxis,:] - min(data_years)))
		  ), [16,25,50,75,84], axis=0)

fig = plt.figure(figsize=(7, 3.5))
ax = plt.gca()
pfb = ax.fill_between(data_years_samp, y2_gp_rho_strong[1], y2_gp_rho_strong[3], color='b', alpha=.25)
pfb2 = ax.fill_between(data_years_samp, y2_gp_rho_strong[0], y2_gp_rho_strong[4], color='b', alpha=.25)
pfg = ax.plot(data_years_samp, y2_gp_rho_strong[2], c='b', lw=2, label='GP model (covariance only)', zorder=0)
ax.axhline(np.exp(stan_model_ext_rho_strong['mu_0'].mean()), color='orange', label='$\mu_0$')

ax.legend(prop={'size':8}, loc=2, ncol=2)
ax.set_ylabel('Annual rate of \npublic mass shootings\n(model)')

ax.set_ylim(0, 2.2)
ax.xaxis.set_minor_locator(FixedLocator(np.arange(min(data_years_samp), max(data_years_samp))))
ax.set_xlim(min(data_years_samp) - 1, max(data_years_samp) + 1)
plt.title(r'Strong prior on $\rho^{-1}$')

plt.savefig('fig_detrended.pdf', bbox_inches='tight')



plot_GP(stan_model_ext_rho_weak)
plt.title(r'Weak prior on $\rho^{-1}$')
plt.savefig('fig_GP_weak.pdf', bbox_inches='tight')




y2_gp_rho_weak = np.percentile(np.exp(
		stan_model_ext_rho_weak['y2'] - 
		np.dot(stan_model_ext_rho_weak['mu_b'][:,np.newaxis], (data_years_samp[np.newaxis,:] - min(data_years)))
		  ), [16,25,50,75,84], axis=0)

fig, axs = plt.subplots(1, figsize=(7,5), sharex='all')
pfb = axs.fill_between(data_years_samp, y2_gp_rho_weak[1], y2_gp_rho_weak[3], color='b', alpha=.25)
pfb2 = axs.fill_between(data_years_samp, y2_gp_rho_weak[0], y2_gp_rho_weak[4], color='b', alpha=.25)
pfg = axs.plot(data_years_samp, y2_gp_rho_weak[2], c='b', lw=2, label='GP model (covariance only)', zorder=0)
axs.axhline(np.exp(stan_model_ext_rho_weak['mu_0'].mean()), color='orange', label='$\mu_0$')

axs.legend(prop={'size':8}, loc=2, ncol=2)
axs.set_ylabel('Annual rate of \npublic mass shootings\n(model)')
axs.set_title(r'Weak $\rho$ prior')

axs.set_ylim(0, 2.2)
axs.xaxis.set_minor_locator(FixedLocator(np.arange(min(data_years_samp), max(data_years_samp))))
axs.set_xlim(min(data_years_samp) - 1, max(data_years_samp) + 1)
plt.title(r'Weak prior on $\rho^{-1}$')
plt.savefig('fig_detrended_weak.pdf', bbox_inches='tight')




#####################################
### Visualize posterior
#####################################


plt.figure()
pa = plt.hist2d(stan_model_ext_rho_strong['mu_0'], 
				stan_model_ext_rho_strong['mu_b'], 
				bins=100, cmap=cm.Reds, cmin=4)
plt.xlabel(r'$\mu_0$ (log shootings)')
plt.ylabel(r'$\mu_b$ (log shootings per year)')
plt.axvline(0, color='k', ls='dashed')
plt.axhline(0, color='k', ls='dashed')
plt.axis([-1.5,1.5,-0.05,.1])
cb = plt.colorbar()
cb.set_label('Number of posterior samples')
plt.title(r'Strong prior on $\rho^{-1}$')
plt.savefig('fig_posterior_2D.pdf', bbox_inches='tight')




## Assemble data matrices
y = pd.Series(stan_model_ext_rho_strong['inv_rho']); y.name = 'inv_rho'
X = pd.DataFrame({
	r'$\eta$':np.sqrt(stan_model_ext_rho_strong['eta_sq']), 
	r'$\mu_0$':stan_model_ext_rho_strong['mu_0'], 
	r'$\mu_b$':stan_model_ext_rho_strong['mu_b'], 
	r'$\sigma$':np.sqrt(stan_model_ext_rho_strong['sigma_sq']), 
	r'$\rm{NB}_{\phi^-1}$':np.sqrt(stan_model_ext_rho_strong['NB_phi_inv']), 
	})
## Standardize
X = X - X.mean()
X = X / X.std()
X = sm.add_constant(X)
X = X.rename(columns={'const':'intercept'})
y = (y - y.mean()) / y.std()
## Fit linear model using statsmodels
est = sm.OLS(y, X).fit()
## Print summary
with open('tab_posterior_regress_strong.tex', 'w') as f:
	c = est.params.index
	v = est.params.values
	d = np.sqrt(np.diag(est.cov_params()))
	s = np.array(['  ',]*len(v))
	for i in range(len(v)):
		if np.abs(v[i] / d[i]) > 2:
			s[i] = '**'
		elif np.abs(v[i] / d[i]) > 1:
			s[i] = '*'
		else: s[i] = ''
	table = '\\\\\n'.join([' & '.join([('%0.3f'%a if type(a)==np.float64 else a) for a in r]) for r in zip(c,v,d,s)])
	f.write(table)



plt.figure()
pa = plt.hist2d(np.sqrt(stan_model_ext_rho_strong['eta_sq']), 
				stan_model_ext_rho_strong['inv_rho'], 
				bins=40, cmap=cm.Reds, cmin=4,
				range = [[0,1],[1,12]])
plt.xlabel(r'$\eta$ (log shootings per year)')
plt.ylabel(r'$\rho^{-1}$ (years)')
sqrt_eta = np.sqrt(stan_model_ext_rho_strong['eta_sq'])
px = np.linspace(min(sqrt_eta), max(sqrt_eta), 10)
px_std = (px - np.mean(sqrt_eta)) / np.std(sqrt_eta)

plt.axis()
cb = plt.colorbar()
cb.set_label('Number of posterior samples')
plt.title(r'Strong prior on $\rho^{-1}$')
plt.savefig('fig_posterior_2D_strong.pdf', bbox_inches='tight')






## Assemble data matrices
y = pd.Series(np.log(stan_model_ext_rho_weak['inv_rho'])); y.name = 'inv_rho'
X = pd.DataFrame({
	r'$\eta$':np.sqrt(stan_model_ext_rho_weak['eta_sq']), 
	r'$\mu_0$':stan_model_ext_rho_weak['mu_0'], 
	r'$\mu_b$':stan_model_ext_rho_weak['mu_b'], 
	r'$\sigma$':np.sqrt(stan_model_ext_rho_weak['sigma_sq']), 
	r'$\rm{NB}_{\phi^-1}$':np.sqrt(stan_model_ext_rho_weak['NB_phi_inv']), 
	})
## Standardize
X = X - X.mean()
X = X / X.std()
X = sm.add_constant(X)
X = X.rename(columns={'const':'intercept'})
y = (y - y.mean()) / y.std()
## Fit linear model using statsmodels
est = sm.OLS(y, X).fit()
# Print summary
with open('tab_posterior_regress_weak.tex', 'w') as f:
	c = est.params.index
	v = est.params.values
	d = np.sqrt(np.diag(est.cov_params()))
	for i in range(len(v)):
		if np.abs(v[i] / d[i]) > 2:
			s[i] = '**'
		elif np.abs(v[i] / d[i]) > 1:
			s[i] = '*'
		else: s[i] = ''
	table = '\\\\\n'.join([' & '.join([('%0.3f'%a if type(a)==np.float64 else a) for a in r]) for r in zip(c,v,d,s)])
	f.write(table)


plt.figure()
pa = plt.hist2d(np.sqrt(stan_model_ext_rho_weak['eta_sq']), 
				stan_model_ext_rho_weak['inv_rho'], 
				bins=40, cmap=cm.Reds, cmin=4,
				range = [[0,4],[1,300]])
plt.xlabel(r'$\eta$ (log shootings per year)')
plt.ylabel(r'$\rho^{-1}$ (years)')
sqrt_eta = np.sqrt(stan_model_ext_rho_weak['eta_sq'])
px = np.linspace(min(sqrt_eta), max(sqrt_eta), 10)
px_std = (px - np.mean(sqrt_eta)) / np.std(sqrt_eta)

plt.axis()
cb = plt.colorbar()
cb.set_label('Number of posterior samples')
plt.title(r'Weak prior on $\rho^{-1}$')
plt.savefig('fig_posterior_2D_weak.pdf', bbox_inches='tight')




#####################################
### Summarize posterior
#####################################


def gt0(y, x, lbound=0, ubound=np.inf):
	y[(x<lbound) & (x>ubound)] = 0
	return y

def marg_post_plot(stan_model_ext, alpha_rho, beta_rho, Nhist=25):
	pdic = {
		'eta_sq': ('$\\eta$', np.sqrt, 'log shootings per year', lambda x: sstats.cauchy.pdf(x**2, 0, 1)),
		'inv_rho': ('$\\rho^{-1}$', lambda x: x, 'years', lambda x: gt0(sstats.gamma.pdf(x, alpha_rho, scale=beta_rho), x, lbound=1)),
		'sigma_sq': ('$\\sigma$', np.sqrt, 'log shootings per year', lambda x: sstats.cauchy.pdf(x**2, 0, 1)),
		'NB_phi_inv':('$\\rm{NB}_{\\phi^{-1}}$', lambda x:x, '', lambda x: sstats.cauchy.pdf(x**2, 0, 0.5)),
		'mu_0': ('$\\mu_0$', lambda x: x, 'log shootings per year, '+str(np.min(data_years)), lambda x: sstats.norm.pdf(x, 0,2)),
		'mu_b': ('$\\mu_b$', lambda x: x, 'annual increase in\nlog shootings per year', lambda x: sstats.norm.pdf(x, 0,0.2)),
		}

	max_x = 3 # maximum number of plots per row
	X = max_x
	Y = len(pdic) / max_x + (np.mod(len(pdic), max_x) > 0)
	fig,axs = plt.subplots(Y, X, figsize=(2.5*X, 2.5*Y), sharey='all')
	axs = axs.flatten()
	for i in range(len(pdic), len(axs)):
		axs[i].set_xticks([])
		axs[i].set_yticks([])
		axs[i].set_visible(0)
		
	fig.text(0.01, 0.5, 'HMC samples ({} total)'.format(N_samp*Nchains), va='center', rotation='vertical')
	for i,hyp in enumerate(pdic.keys()):
		samps = pdic[hyp][1](stan_model_ext[hyp])
		hn, hb, hp = axs[i].hist(samps, Nhist, edgecolor='none', facecolor='.5', label='Posterior samples')
		ppx = np.linspace(np.min(samps), np.max(samps), 10000)
		ppy = pdic[hyp][1]( pdic[hyp][3](ppx) )
		## Normalize
		ppy *= len(samps) / np.sum(ppy) * len(ppy) / len(hn)
		axs[i].plot(ppx, ppy, color='b', zorder=2, label='Hyperprior')
		axs[i].xaxis.set_major_locator(MaxNLocator(3))
		axs[i].xaxis.set_minor_locator(AutoMinorLocator(3))
		axs[i].set_xlabel(pdic[hyp][0] + ' ({})'.format(pdic[hyp][2]), ha='center', size=9)
		axs[i].set_ylim(0, 10900)
		axs[i].axvline(0, ls='dashed', color='.2')
	axs[X-1].legend(prop={'size':9}, bbox_to_anchor=(0, 1.4))
	plt.subplots_adjust(hspace=0.5, wspace=0.5)
	
	return fig, axs


fig, axs = marg_post_plot(stan_model_ext_rho_strong, stan_data_rho_strong['alpha_rho'], 1/stan_data_rho_strong['beta_rho'], Nhist=100)
axs[0].set_title(r'Strong prior on $\rho^{-1}$')
plt.savefig('fig_posterior_1D_strong.pdf', bbox_inches='tight')

	

fig, axs = marg_post_plot(stan_model_ext_rho_weak, stan_data_rho_weak['alpha_rho'], 1/stan_data_rho_weak['beta_rho'], Nhist=100)
axs[0].set_title(r'Weak prior on $\rho^{-1}$')
plt.savefig('fig_posterior_1D_weak.pdf', bbox_inches='tight')




## Write out some facts for each model
print_ext_names = ['strongprior', 'weakprior']

## Probability that mu_b is positive
with open(fact_file, 'a') as f:
	for i in range(2):
		f.write("\\newcommand{\\"+print_ext_names[i]+"mubgtrzero}{%0.0f"%(100 * np.mean(exts[i]['mu_b'] > 0))+"}\n")

zincreaseraw = {}
with open(fact_file, 'a') as f:
	for i in range(2):
		zincreaseraw[i] = (np.exp((max_year - np.min(data_years)) * exts[i]['mu_b']) - 1) * 100
		zincrease = np.percentile(zincreaseraw[i], [16,50,84])
		z_str = '%0.0f'%round(zincrease[1], -1)+'~[%0.0f'%round(zincrease[0], -1)+',~%0.0f'%round(zincrease[2], -1)+']'
		f.write("\\newcommand{\\"+print_ext_names[i]+"zincrease}{"+z_str+"}\n")
		f.write("\\newcommand{\\"+print_ext_names[i]+"zincreasegtrpop}{%0.0f"%(np.mean(zincreaseraw[i] > (323/231. - 1)*100)*100)+"}\n")

i1 = np.argmin(abs(data_years_samp - 2011.5))
i2 = np.argmin(abs(data_years_samp - 2014.5))
py = np.exp(stan_model_ext_rho_strong['y2'][:,i2]) / np.exp(stan_model_ext_rho_strong['y2'][:,i1])

plt.figure()
ph = plt.hist(py, 50, edgecolor='none', facecolor='.8', range=[0,8], normed=1)
plt.xlabel('Relative rate of public mass shootings in 2014 versus 2011')
plt.ylabel('Posterior probability')
plt.axvline(1, color='k', label='Unity')
plt.axvline(np.mean(py), color='0.3', label='Mean posterior estimate', ls='dashed')
plt.axvline(3, color='g', label='Cohen et al. estimate', lw=2, ls='dotted')
plt.legend()
plt.title(r'Strong prior on $\rho^{-1}$')
plt.savefig('fig_rate_increase.pdf', bbox_inches='tight')


with open(fact_file, 'a') as f:
	f.write("\\newcommand{\\strongpriorprobrateincrease}{%0.0f"%(np.mean(py > 1) * 100)+"}\n")
	f.write("\\newcommand{\\strongpriormeanincrease}{%0.1f"%(np.mean(py))+"}\n")
	f.write("\\newcommand{\\strongpriormeanincreasepercent}{%0.0f"%((np.mean(py)-1) * 100)+"}\n")
	f.write("\\newcommand{\\strongpriorprobthreeXincrease}{%0.0f"%(np.mean(py > 3) * 100)+"}\n")
	
	typ_uncertain_strong = np.mean(y2_gp_rho_strong[3] - y2_gp_rho_strong[1])
	typ_uncertain_weak = np.mean(y2_gp_rho_weak[3] - y2_gp_rho_weak[1])
	f.write("\\newcommand{\\uncertaintyratiostrongweak}{%0.0f"%((typ_uncertain_strong / typ_uncertain_weak - 1) * 100)+"}\n")
