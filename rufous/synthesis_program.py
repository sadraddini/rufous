#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 16:39:18 2020

@author: sadra
"""

import numpy as np
import pickle

try:
    import pydrake.solvers.mathematicalprogram as MP
    import pydrake.solvers.gurobi as Gurobi_drake
    global gurobi_solver,OSQP_solver
    gurobi_solver=Gurobi_drake.GurobiSolver()
    import pydrake.symbolic as sym
except:
    print("Error in loading Drake Mathematical Program")

