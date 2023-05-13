import numpy as np
import pandas as pd
import seaborn as sn

from sklearn.ensemble import RandomForestRegressor as rfr

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV as rscv
from sklearn.model_selection import GridSearchCV as gscv
from sklearn.model_selection import cross_val_score as cvs

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE

from scipy.stats import randint

from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import max_error, r2_score

pip install -U pymoo;

rs = 3851

def score(model, X:list, y:list):
  print(model,'\n')
  res = model.predict(X[1])
  #Scores
  print('Score on test Values:')
  print('R Squared Value:', r2_score(res, y[1]))
  print('Mean absolute percentage error:', mape(res, y[1]))
  print('Maximum error:', max_error(res, y[1]))
  print('')
  kf = KFold(n_splits = 5, shuffle=False)
  cvr = cross_validate(model, X[0], y[0], cv=kf, scoring = ['r2', 'neg_mean_absolute_percentage_error', 'max_error'], n_jobs=-1)
  print('Cross Validate:')
  print('R Squared Mean and Standard deviation:',cvr['test_r2'], cvr['test_r2'].mean(), cvr['test_r2'].std())
  print('Mean absolute percentage error Mean and Standard deviation:',cvr['test_neg_mean_absolute_percentage_error'], cvr['test_neg_mean_absolute_percentage_error'].mean(), cvr['test_neg_mean_absolute_percentage_error'].std())
  print('Maximum error Mean and Standard deviation:',cvr['test_max_error'], cvr['test_max_error'].mean(), cvr['test_max_error'].std())

def tune_rf_model(model, X_train, y_train):

  kf = KFold(n_splits = 5, shuffle=False)

  max_depth = [i for i in range(4,9)]
  n_estimators = [i for i in range(7,15)]
  min_samples_split = [i for i in range(4, 9)]
  min_samples_leaf = [i for i in range(3, 7)]

  rf_params = {
      'max_depth':max_depth,
      'n_estimators': n_estimators,
      'min_samples_split':min_samples_split,
      'min_samples_leaf':min_samples_leaf,
  }

  rscvr = rscv(model, rf_params, cv = kf,n_iter = 200, scoring='max_error', n_jobs = -1)
  search = rscvr.fit(X_train, y_train)
  searchd = search.best_params_
  print(searchd)

  rfn = rfr(
    max_depth=searchd['max_depth'], 
    n_estimators=searchd['n_estimators'], 
    min_samples_split=searchd['min_samples_split'],
    min_samples_leaf=searchd['min_samples_leaf'],
    n_jobs=-1, random_state = rs)
  
  return rfn

def pred_val(model1, model2):
  val1 = model1.predict(X)
  print("val1", val1)
  min_dev_1 = min(abs(val1-target_value_1))
  idx = list(abs(val1-target_value_1)).index(min_dev_1)
  print(idx)
  print(val1[idx])
  print("")

  val2 = model2.predict(X)
  print("val2", val2)

  min_dev_2 = min(abs(val2-target_value_2))
  idx_2 = list(abs(val2-target_value_2)).index(min_dev_2)
  print(idx_2)
  print(val2[idx_2])
  print("")

  r = np.sqrt((1 - val1/target_value_1)**2 + (1 - val2/target_value_2)**2)

  idx_min = list(r).index(min(r))
  print(idx_min)
  print(min(r))
  print(X[idx_min])

  return X[idx_min]

dat_m = pd.read_csv('/content/Test data 2.csv')
dat_m.tail(5)

out = ['DRCM']
inp = ['CaO',	'SiO',	'H20', 'FA', 'CA', 'Alk', 'Total Binder']

n_e = len(inp)

Xm = dat_m[inp]
ym = dat_m[out]

X_trn_m, X_tst_m, y_trn_m, y_tst_m = train_test_split(Xm, ym, random_state = rs)

Xml = [X_trn_m, X_tst_m]
yml = [y_trn_m, y_tst_m]

rfm = rfr(random_state=rs)
rfm.fit(Xml[0], yml[0])
score(rfm, Xml, yml)

tune_rf_model(rfm, Xml[0], yml[0])

rfmn = rfr(
    max_depth=9,#6,
    n_estimators=8,#8,
    min_samples_split=7,#6,
    min_samples_leaf=2,#4,
    n_jobs=-1, random_state = rs)
rfmn.fit(Xml[0], yml[0])
score(rfmn, Xml, yml)

x = [149.124553,128.238074,	205.0,	743.146756,	975.0,	359.649123]
rfmn.predict([x])

dat_s = pd.read_csv('/content/Data strnth 28 day.csv')
dat_s.tail(1)

out_s = ['Strength']
#inp_s = ['Cao',	'SiO2',	'H20', 'FA', 'CA', 'Total Binder','Time/Strength']
inp_s = ['Cao',	'SiO2',	'H20/B', 'FA', 'CA', 'Total Binder']

Xs = dat_s[inp_s]
ys = dat_s[out_s]

X_trn_s, X_tst_s, y_trn_s, y_tst_s = train_test_split(Xs, ys, random_state = rs)

Xsl = [X_trn_s, X_tst_s]
ysl = [y_trn_s, y_tst_s]

rfs = rfr(random_state=rs)
rfs.fit(Xsl[0], ysl[0])
score(rfs, Xsl, ysl)

tune_rf_model(rfs, Xsl[0], ysl[0])

rfsn = rfr(
    max_depth=6,#6,
    n_estimators=28,#8,
    min_samples_split=2,#4,
    min_samples_leaf=1,#5,
    n_jobs=-1, random_state = rs)
rfsn.fit(Xsl[0], ysl[0])
score(rfsn, Xsl, ysl)

x = [149.124553,128.238074,	205.0,	743.146756,	975.0,	359.649123]
rfsn.predict([x])

x =[229.967578,	123.0095,	196.1415,	728.96,	1110.16,	435.87]
rfsn.predict([x])

228.0119	127.8334	205	0.46	672.1601	975	445.6521739

def function_1(model1, target_value_1, X):
  pred1 = abs(model1.predict([X]) - (1-0.02)*target_value_1)
  return pred1

def function_2(model2, target_value_2, X):
  pred2 = abs(model2.predict([X]) - (1+0.02)*target_value_2)
  return pred2

target_value_1 = 2.95

target_value_2 = 42.9

dat_s = pd.read_csv('/content/Data strnth 28 day.csv')
dat_s.head(1)

from pymoo.core.problem import ElementwiseProblem

class functions(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=n_e,
                         n_obj=2,
                         n_ieq_constr=0,
                         xl=np.array([100,0,0.35,400, 900, 400]),
                         xu=np.array([400, 200, 0.57, 800, 1200, 600]))

    def _evaluate(self, x, out, *args, **kwargs):

        Xt = []

        for i in range(n_e):
          Xt.append(x[i])
     
        f1 = function_1(rfmn, target_value_1, Xt)
        f2 = function_2(rfsn, target_value_1, Xt)

        out["F"] = [f1, f2]

problem = functions()

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

algorithm = NSGA2(
    pop_size=100,
    n_offsprings=40,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.7, eta=10),
    mutation=PM(eta=10),
    eliminate_duplicates=True
)

from pymoo.termination import get_termination

termination = get_termination("n_gen", 100)

from pymoo.optimize import minimize

res = minimize(problem,
               algorithm,
               termination,
               seed=None,
               save_history=True,
               verbose=True)

X = res.X
F = res.F

X

pred_val(rfmn, rfsn)

