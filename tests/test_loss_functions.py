#python -m scoring_rules_paper_experiments.tests.test_loss_functions
from ..loss_functions import CostSensitiveLoss, CostSensitiveLossWithRejectOption, BrierLoss, LogLoss
import matplotlib.pyplot as plt
import numpy as np

costSensitive = CostSensitiveLoss(c=0.2)
brier = BrierLoss()
log = LogLoss()


"""
Test cases
"""

p,q = (0.3,0.7)
assert costSensitive.getMinMaxAction(p,q) == 1.
assert brier.getMinMaxAction(p,q) == 0.5
assert log.getMinMaxAction(p,q) == 0.5

p,q = (0.1,0.21)
assert costSensitive.getMinMaxAction(p,q) == 0.
assert brier.getMinMaxAction(p,q) == 0.21
assert log.getMinMaxAction(p,q) == 0.21

p,q = (0.,0.1)
assert costSensitive.getMinMaxAction(p,q) == 0.
assert brier.getMinMaxAction(p,q) == 0.1
assert log.getMinMaxAction(p,q) == 0.1

p,q = (0.7,0.99)
assert costSensitive.getMinMaxAction(p,q) == 1.
assert brier.getMinMaxAction(p,q) == 0.7
assert log.getMinMaxAction(p,q) == 0.7

print("all tests passed")


# test the cost-sensitive with reject option loss function
crloss = CostSensitiveLossWithRejectOption(c=0.2,r=0.1)
xs = np.linspace(0.0,1.0,500)
plt.figure()
plt.title("Entropy of %s" % crloss.name)
plt.plot(xs, crloss.entropy(xs))
plt.show()