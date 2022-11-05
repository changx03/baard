"""Constants for attacks."""
from .apgd import auto_projected_gradient_descent
from .cw2 import carlini_wagner_l2
from .fast_gradient_method import fast_gradient_method
from .projected_gradient_descent import projected_gradient_descent

ATTACKS = ['FGSM', 'PGD', 'CW2', 'APGD']
L_NORM = ['inf', 2]
