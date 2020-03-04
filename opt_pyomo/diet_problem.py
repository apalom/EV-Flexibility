# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:52:04 2020
@author: Danzig (1990)

https://nbviewer.jupyter.org/github/Pyomo/PyomoGallery/blob/master/diet/DietProblem.ipynb

The Diet Problem can be formulated mathematically as a linear programming problem using the following model.

Sets
F = set of foods
N = set of nutrients

Parameters
ci = cost per serving of food i, ∀i∈F
aij = amount of nutrient j in food i, ∀i∈F,∀j∈N
Nminj = minimum level of nutrient j, ∀j∈N
Nmaxj = maximum level of nutrient j, ∀j∈N
Vi = the volume per serving of food i, ∀i∈F
Vmax = maximum volume of food consumed

Variables
xi = number of servings of food i to consume

Objective
Minimize the total cost of the food
min∑i∈Fcixi
Constraints
Limit nutrient consumption for each nutrient j∈N.
Nminj≤∑i∈Faijxi≤Nmaxj, ∀j∈N
Limit the volume of food consumed
∑i∈FVixi≤Vmax
Consumption lower bound
xi≥0, ∀i∈F
"""

# We begin by importing the Pyomo package and creating a model object:
from pyomo.environ import *
infinity = float('inf')

model = AbstractModel()

#%---Iniialize sets---

# Foods
model.F = Set()
# Nutrients
model.N = Set()

#%---Define model parameters---
# The within option is used in these parameter declarations to define expected 
# properties of the parameters. This information is used to perform error checks 
# on the data that is used to initialize the parameter components.

# Cost of each food
model.c    = Param(model.F, within=PositiveReals)
# Amount of nutrient in each food
model.a    = Param(model.F, model.N, within=NonNegativeReals)
# Lower and upper bound on each nutrient
model.Nmin = Param(model.N, within=NonNegativeReals, default=0.0)
model.Nmax = Param(model.N, within=NonNegativeReals, default=infinity)
# Volume per serving of food
model.V    = Param(model.F, within=PositiveReals)
# Maximum volume of food consumed
model.Vmax = Param(within=PositiveReals)

#%---Define model decision variables---
# Number of servings consumed of each food
model.x = Var(model.F, within=NonNegativeIntegers)

#%---Define objective function---
# The Objective component is used to define the cost objective. This component 
# uses a rule function to construct the objective expression:

# Minimize the cost of food that is consumed
def cost_rule(model):
    return sum(model.c[i]*model.x[i] for i in model.F)
model.cost = Objective(rule=cost_rule)

#%---Define constraints---
# Similarly, rule functions are used to define constraint expressions 
# in the Constraint component:

# Limit nutrient consumption for each nutrient
def nutrient_rule(model, j):
    value = sum(model.a[i,j]*model.x[i] for i in model.F)
    return model.Nmin[j] <= value <= model.Nmax[j]
model.nutrient_limit = Constraint(model.N, rule=nutrient_rule)

# Limit the volume of food consumed
def volume_rule(model):
    return sum(model.V[i]*model.x[i] for i in model.F) <= model.Vmax
model.volume = Constraint(rule=volume_rule)


#%% Define Model Data