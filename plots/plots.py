# other helper plots for the paper
#python -m ip_scoring_rules_experiments.plots.plots
# execute from directory level above the ip_scoring_rules_experiments folder
from ..loss_functions import CostSensitiveLoss, AsymmetricLoss, AsymmetricScore
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

plotdir = "ip_scoring_rules_experiments/plots/"

ps = np.linspace(1e-6,1.0-1e-6,1000)
c = 0.1
closs = CostSensitiveLoss(c)
fac = 500.
aloss_smooth = AsymmetricLoss(c,smooth=True,smoothing_factor=fac)
aloss_unsmooth = AsymmetricLoss(c,smooth=False)


palette = sns.color_palette()



def T(c,a):
	if a>=c:
		return -(c-1)**2
	else:
		return -c**2

def smooth_T(c, r, smoothing_factor=1000.):
    smooth_step = torch.sigmoid(smoothing_factor * (r - c))  # Sigmoid approximation
    return smooth_step * (-(c-1)**2) + (1 - smooth_step) * (-c**2)

fs = 1.5
with sns.plotting_context('notebook',font_scale=fs): 
	fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
	ax.plot(ps, [T(c,p) for p in ps],color=palette[0],label=r"Step function $T(c,a)$")

	# smooth step
	ps = torch.linspace(1e-6,1.0-1e-6,1000)
	ax.plot(ps, smooth_T(c,ps,smoothing_factor=fac),color=palette[1],label=r"Smooth approximation $\tilde{T}(c,a)$")
	ax.set_xlabel("a")
	ax.legend(loc=0)
	plt.savefig(plotdir+"step_function_approximation.pdf",bbox_inches='tight')
	plt.show()



# plot the asymmetric loss function for fixed outcome
fs = 1.5
with sns.plotting_context('notebook',font_scale=fs): 
	fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
	#plt.plot(ps,aloss_smooth.ell(ps, np.zeros(len(ps))),color=palette[0],label=r"$\ell_{W;%s}(a,y=0)$ (smooth approximation)"%c)
	#plt.plot(ps,aloss_smooth.ell(ps, np.ones(len(ps))),color=palette[1],label=r"$\ell_{W;%s}(a,y=1)$ (smooth approximation)"%c)
	ax.plot(ps,aloss_unsmooth.ell(ps, np.zeros(len(ps))),color=palette[2],label=r"$\ell_{W;%s}(a,y=0)$"%c)
	ax.plot(ps,aloss_unsmooth.ell(ps, np.ones(len(ps))),color=palette[3],label=r"$\ell_{W;%s}(a,y=1)$"%c)
	ax.set_xlabel("a")
	ax.legend(loc=0)
	plt.savefig(plotdir+"asymmetric_loss.pdf",bbox_inches='tight')
	plt.show()

fs = 1.5
with sns.plotting_context('notebook',font_scale=fs): 
	ps_=np.linspace(0.08,0.12,1000)
	fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
	ax.plot(ps_,aloss_smooth.ell(ps_, np.zeros(len(ps_))),color=palette[0],label=r"Approx. of $\ell_{W;%s}(a,y=0)$"%c)
	ax.plot(ps_,aloss_smooth.ell(ps_, np.ones(len(ps_))),color=palette[1],label=r"Approx. of $\ell_{W;%s}(a,y=1)$"%c)
	ax.plot(ps_,aloss_unsmooth.ell(ps_, np.zeros(len(ps_))),color=palette[2],label=r"$\ell_{W;%s}(a,y=0)$"%c)
	ax.plot(ps_,aloss_unsmooth.ell(ps_, np.ones(len(ps_))),color=palette[3],label=r"$\ell_{W;%s}(a,y=1)$"%c)
	ax.set_xlabel("a")
	ax.legend(loc="upper right")
	plt.savefig(plotdir+"asymmetric_loss_approximation.pdf",bbox_inches='tight')
	plt.show()


clossvals = closs.entropy(ps)
alossvals_smooth = aloss_smooth.entropy(ps)
alossvals_unsmooth = aloss_unsmooth.entropy(ps)

clossvals /= max(clossvals)
alossvals_smooth /= max(alossvals_unsmooth) 
alossvals_unsmooth /= max(alossvals_unsmooth) 



fs = 1.5
with sns.plotting_context('notebook',font_scale=fs): 
	plt.figure()
	plt.plot(ps, clossvals, color=palette[0], label=r"Entropy of ${\ell_{%s}}$"%c)
	plt.plot(ps,alossvals_unsmooth, color=palette[1], label=r"Entropy of ${\ell_{W;%s}}$"%c)
	plt.plot(ps,alossvals_smooth, color=palette[1], label=r"Entropy of smooth ${\ell_{W;%s}}$"%c)
	plt.axvline(x=c, alpha=.3, linestyle='--', color='k')
	plt.legend(loc=0)
	plt.xlabel(r"$\Delta^2$")
	plt.savefig(plotdir + "entropy_comparison_asymmetric.pdf",bbox_inches='tight')
	plt.show()