# -*- coding: utf-8 -*-
"""Reverse Learning For Concrete in TensorFlow

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bX-n6RBbC9PyBOQx9Ly-HLtHz0lPA1zv
"""

pip install tensorflow_decision_forests

pip install -U pymoo

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_decision_forests as tfdf

rs = 5138
np.random.seed(rs)

data = pd.read_csv('/content/td2.csv')
data.head(2)

data.describe()

data1 = data[['CaO', 'SiO', 'Total Binder', 'H20','FA', 'CA', 'Alk', 'DRCM']]
data2 = data[['CaO', 'SiO', 'Total Binder', 'H20','FA', 'CA', 'Alk','Strength']]
features = ['CaO', 'SiO', 'Total Binder', 'H20','FA', 'CA', 'Alk']
n_f = len(features)

def f_val_min_max(data, features:list, change_pc=0):
  n_f = len(features)
  min_reduce = 1-change_pc/100
  max_increase = 1+change_pc/100
  min_vals = []
  max_vals = []
  for i in range(n_f):
    min_vals.append(min_reduce*min(data[features[i]]))
    max_vals.append(max_increase*max(data[features[i]]))

  return min_vals, max_vals

def split_dataset(dataset, test_ratio=0.20):
  """Splits a panda dataframe in two."""
  test_indices = np.random.rand(len(dataset)) < test_ratio
  return dataset[~test_indices], dataset[test_indices]

train1, test1 = split_dataset(data1)
train2, test2 = split_dataset(data2)

D_train = tfdf.keras.pd_dataframe_to_tf_dataset(train1, label="DRCM", task = tfdf.keras.Task.REGRESSION)
D_test = tfdf.keras.pd_dataframe_to_tf_dataset(test1, label="DRCM", task = tfdf.keras.Task.REGRESSION)

S_train = tfdf.keras.pd_dataframe_to_tf_dataset(train2, label="Strength", task = tfdf.keras.Task.REGRESSION)
S_test = tfdf.keras.pd_dataframe_to_tf_dataset(test2, label="Strength", task = tfdf.keras.Task.REGRESSION)

tunerd = tfdf.tuner.RandomSearch(num_trials=int((51+38)), use_predefined_hps=True, trial_num_threads=10)  #*((3*8)/(5*1))
tuners = tfdf.tuner.RandomSearch(num_trials=int((51+38)*((1+8)/(5+3))), use_predefined_hps=True, trial_num_threads=10)

"""Diffusion Coefficient"""

rf1t = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION, random_seed = rs, tuner = tunerd, verbose=0)
rf1t.compile()
rf1t.fit(D_train)
rf1t.compile(metrics=['mse'])
e = rf1t.evaluate(D_test, return_dict=True)
print(e)
print(e['mse'])
print(np.sqrt(e['mse']))

r = rf1t.predict(D_test)
s = np.array(test1["DRCM"])
l = []
for i in r:
  l.append(float(i))

abs(s - np.array(l))/s*100

"""Strength"""

rf2t = tfdf.keras.RandomForestModel(task = tfdf.keras.Task.REGRESSION, random_seed = rs, tuner = tuners, verbose=0)
rf2t.fit(S_train)
rf2t.compile(metrics=['mse'])
e = rf2t.evaluate(S_test, return_dict=True)
print(e)
print(e['mse'])
print(np.sqrt(e['mse']))

r = rf2t.predict(S_test)
s = np.array(test2["Strength"])
l = []
for i in r:
  l.append(float(i))

abs(s - np.array(l))/s*100

"""Find design"""

def dif_cof(model, X):
  tp = pd.DataFrame(X, columns = features)
  t = tfdf.keras.pd_dataframe_to_tf_dataset(tp)
  r = model.predict(t)
  return r

def strength(model, X):
  tp = pd.DataFrame(X, columns = features)
  t = tfdf.keras.pd_dataframe_to_tf_dataset(tp)
  r = model.predict(t)
  return r

X = [3,0,0,0,0,0,0]

strength(rf2t, [X])

def mass_of_material(M_CaO, M_SiO2, M_Binder, fa=True, s=True, cond_min=False):

  if fa==False:
    mass = [M_CaO, M_SiO2]
    Cmax = [[0.673299, 0.490982],[0.490982, 0.313951]]   #Cement, Slag
  elif s==False:
    mass = [M_CaO, M_SiO2]
    Cmax = [[0.673299, 0.026649],[0.170648, 0.635439]]   #Cement, Flyash
  else:
    mass = [M_CaO, M_SiO2, M_Binder]
    Cmax = [[0.673299, 0.026649, 0.490982],[0.170648, 0.635439, 0.313951], [1, 1, 1]]   #Cement, Flyash, Slag


  mass_m = np.linalg.solve(Cmax, mass)
  return mass_m

def alkalinity(M_CaO, M_SiO2, M_Binder, fa=True, s=True, cond_min=False):
  if fa==False:
    mses = mass_of_material(M_CaO, M_SiO2, M_Binder, fa, s, cond_min) #cement, slag
    alk = np.array([1.7686, 9.165])*mses[0] + np.array([4.4165, 15.1393])*mses[1]

  elif s==False:
    mses = mass_of_material(M_CaO, M_SiO2, M_Binder, fa, s, cond_min) #cement, flyash
    alk = np.array([1.7686, 9.165])*mses[0] + np.array([6.4208, 24.47])*mses[1]

  else:
    mses = mass_of_material(M_CaO, M_SiO2, M_Binder, fa, s, cond_min) #cement, flyash, slag
    alk = np.array([1.7686, 9.165])*mses[0] + np.array([6.4208, 24.47])*mses[1] + np.array([4.4165, 15.1393])*mses[2]

  return alk

tdc = 8
tstr = 40

from pymoo.core.problem import ElementwiseProblem

class functions(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=n_f,
                         n_obj=2,
                         n_ieq_constr=0,
                         #n_eq_constr=1,
                         #["Cao","SiO2","H20",'Alk', "Total Binder","FA","CA"]
                         ##['CaO', 'SiO', 'Total Binder', 'H20','FA', 'CA', 'Alk']
                         #xl=np.array([100, 50, 500, 205, 500, 950, 15,]),
                         #xu=np.array([400, 200, 650, 220, 900, 1200, 29])
                         xl = f_val_min_max(data, features, change_pc=2)[0],
                         xu = f_val_min_max(data, features, change_pc=2)[1]
                         )


    def _evaluate(self, x, out, *args, **kwargs):

        Xt = []

        for i in range(n_f):
          Xt.append(x[i])

        f1 = abs(dif_cof(rf1t, [Xt]) - tdc)
        f2 = abs(strength(rf2t, [Xt]) - tstr)

        #h1 = np.sum(mass_of_material(x[0], x[1], fa=True, s=False, cond_min=False)) - x[2]

        out["F"] = [f1, f2]
        #out["H"] = [h1]

problem = functions()

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

algorithm = NSGA2(
    pop_size=75,
    n_offsprings=30,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.7, eta=5),
    mutation=PM(eta=7),
    eliminate_duplicates=True
)

from pymoo.termination import get_termination

termination = get_termination("n_gen", 20)

from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               termination,
               seed=rs,
               save_history=True,
               verbose=True)

X = res.X
F = res.F

X

F

def dif_cof(model, X):
  tp = pd.DataFrame(X, columns = features)
  t = tfdf.keras.pd_dataframe_to_tf_dataset(tp)
  r = model.predict(t)
  return r

def strength(model, X):
  tp = pd.DataFrame(X, columns = features)
  t = tfdf.keras.pd_dataframe_to_tf_dataset(tp)
  r = model.predict(t)
  return r

def dist_cof(model1, model2, X):
  tp = pd.DataFrame(X, columns = features)
  t = tfdf.keras.pd_dataframe_to_tf_dataset(tp)
  r1 = model1.predict(t)
  r2 = model2.predict(t)
  r = list(np.sqrt((1-r1/tdc)**2 + (1-r2/tstr)**2))
  rmin_i = r.index(min(r))

  return X[rmin_i]

def dist(F):
  n=len(F)
  f = list(F)
  err = 5138
  erl = []
  for i in range(n):
    err1 = f[i][1]
    if err1<=err and f[i][0]>0.5:
      err = err1
    else:
      err = err
    erl.append(err)

  rmin = erl.index(err)
  return rmin

X[int(dist(F))]

v = (X[dist(F)])

#v = dist_cof(rf1t, rf2t, X)
v2 = []
for i in v:
  v2.append(int(i))

v2

dif_cof(rf1t, [list(v)])

strength(rf2t, [list(v)])

a = [256.247633, 108.052433, 435.87, 196.1415, 692.93, 1110.16, 24.82823766]
b = [257.93277269,  171.9759386 ,  437.97998386,  201.69973688, 626.68504044, 1117.74710378,   48.14336098]

dif_cof(rf1t, [list(b)])

strength(rf2t, [list(b)])

