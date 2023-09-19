#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sn
import pickle as pk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
import threading
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor as rfr

rs = 5138

#data = pd.read_csv('/content/TestData.csv')
#data.head(2)

global tdc, tstr

xmin = [1.461376e+02, 6.014260e+01, 2.744000e-01, 1.901200e+00, 1.922172e+02, 5.286806e+02, 9.459352e+02, 3.524570e+02]
xmax = [ 428.5122,  210.6606,    1.4994,   11.1078,  229.5306,  778.515, 1135.617,  653.3202]

def fraction_df(data):
  datfr = pd.DataFrame()
  datfr['C/B'] = data['CaO']/data['Total Binder']
  datfr['S/B'] = data['SiO']/data['Total Binder']
  datfr['H/B'] = data['H20']/data['Total Binder']
  datfr['A/B'] = (data['Na2O']+data['K2O'])/data['Total Binder']
  datfr['F/T'] = data['FA']/(data['FA']+data['CA'])
  datfr['C/T'] = data['CA']/(data['FA']+data['CA'])

  return datfr


def f_val_min_max(data, features:list, change_pc=0):
  '''function to get the minimum and maximum value of a feature'''
  n_f = len(features)
  min_reduce = 1-change_pc/100
  max_increase = 1+change_pc/100
  min_vals = []
  max_vals = []
  for i in range(n_f):
    min_vals.append(min_reduce*min(data[features[i]]))
    max_vals.append(max_increase*max(data[features[i]]))

  return min_vals, max_vals

features = ['CaO', 'SiO', 'Na2O', 'K2O', 'H20', 'FA', 'CA', 'Total Binder']
n_f = len(features)


#Diffusion Coefficient Train
rfr_d = pk.load(open('RFR_drcm.sav', 'rb'))

#Strength Train
rfr_s = pk.load(open('RFR_strngth.sav', 'rb'))

def predict(model, X):
  td = pd.DataFrame(X, columns = features)
  tp = fraction_df(td)
  r = model.predict(tp)
  return r


def mass_of_material(M_CaO, M_SiO2, M_K2O, fa=True, s=True):

  if fa==False:
    mass = [M_CaO, M_SiO2]
    Cmax = [np.array([65.59, 48.59])/100, np.array([19.59, 32.59])/100]   #Cement, Slag
  elif s==False:
    mass = [M_CaO, M_SiO2]
    Cmax = ([np.array([65.59, 1.44])/100, np.array([19.59, 62.75])/100])   #Cement, Flyash
  else:
    mass = [M_CaO, M_SiO2, M_K2O]
    Cmax = [np.array([65.59, 1.44, 48.59])/100, np.array([19.59, 62.75, 32.59])/100, np.array([0.57, 2.65, 0.38])/100]   #Cement, Flyash, Slag

  mass_m = np.linalg.solve(Cmax, mass)
  return mass_m


def design(tstr, tdc):
    from pymoo.core.problem import ElementwiseProblem

    class functions(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=n_f,
                             n_obj=2,
                             n_ieq_constr=1,
                             n_eq_constr=0,
                             #'CaO', 'SiO', 'Na2O', 'K2O', 'H20', 'FA', 'CA', 'Total Binder'
                             #xl = f_val_min_max(data, features, change_pc=2)[0],
                             #xu = f_val_min_max(data, features, change_pc=2)[1]
                             xl = xmin,
                             xu = xmax
                             )


        def _evaluate(self, x, out, *args, **kwargs):

            Xt = []

            for i in range(n_f):
              Xt.append(x[i])

            f1 = abs(predict(rfr_d, [Xt]) - tdc)
            f2 = abs(predict(rfr_s, [Xt]) - tstr)

            g1 = x[7] - np.sum(mass_of_material(x[0], x[1], x[3], fa, s))
            #g2 = -(x[4]/x[7] - 0.99*wc_ratio)
            #g3 = (x[4]/x[7] - 1.01*wc_ratio)

            out["F"] = [f1, f2]
            out["G"] = [g1]#g2, g3]


    problem = functions()

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling

    algorithm = NSGA2(
        pop_size=50,
        n_offsprings=30,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.7, eta=5),
        mutation=PM(eta=5),
        eliminate_duplicates=True
    )

    from pymoo.termination import get_termination

    termination = get_termination("n_gen", 50)#50)

    from pymoo.optimize import minimize

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=rs,
                   save_history=True,
                   verbose=True)

    X = res.X
    F = res.F
    return F, X

def select(F, X):
    x = [i[0] for i in F]
    y = [i[1] for i in F]
    close_val = abs(45 - np.arctan((np.array(y)/tstr)/(np.array(x)/tdc))*360/np.pi/2)
    close_val = list(close_val)
    indx = close_val.index(min(close_val))
    return X[indx], indx

    #['CaO', 'SiO', 'Na2O', 'K2O', 'H20', 'FA', 'CA', 'Total Binder']
    #materials = select(F, X)

    #mass_of_material(materials[0], materials[1], materials[3], fa, s)
    
    
root = Tk()
root.title("Concrete Designer")
root.iconbitmap()
root.geometry("450x450")
root.resizable(False, False)

'''Extras'''
global progbar
progbar = ttk.Progressbar(
    root,
    orient='horizontal',
    mode='indeterminate',
    length=350
)

progbar.grid(row = 3, column = 0, columnspan=2, padx=10, pady=20)

'''End of Extras'''


'''Supplementary material selection'''
frameRad = ttk.LabelFrame(root, text="Supplementary Materials")#, padx=5, pady=5)
frameRad.grid(row = 0, column=0, padx=5, pady=5)

r = IntVar()
r.set(1)
#global fa, s, supc
fa = True
s = False
supc = "Fly Ash"

def rad_val(value):
    global fa, s, supc
    if value==1:
        fa = True #True #Fly ash
        s = False #False #Slag
        supc = "Fly Ash"
        
    elif value==2:
        fa = False #True #Fly ash
        s = True #False #Slag
        supc = "Slag"
    
    Label_text_init = ('Cement:  '+'0.00'+'\n\n'+
                           supc+': 0.00'+'\n\n'
                           'Water:  '+'0.00'+'\n\n'
                           'Fine Aggregates:  '+'0.00'+'\n\n'
                           'Coarse Aggregates:  '+'0.00')
    labr.config(text=Label_text_init)
    

rb1 = ttk.Radiobutton(frameRad, text="Fly Ash", variable=r, value=1, command=lambda: rad_val(r.get()))
rb1.pack()  
rb2 = ttk.Radiobutton(frameRad, text="Slag", variable=r, value=2, command=lambda: rad_val(r.get()))
rb2.pack()
'''End of Supplementary material selection'''


'''Estimation of Design'''

frame = ttk.LabelFrame(root, text="Input Data")#, padx=5, pady=5)
frame.grid(row = 0, column =1, padx=5, pady=5)

ent1 = ttk.Entry(frame, width=15)
ent1.grid(row=0, column=1, padx= 5, pady = 5)
l1 = ttk.Label(frame, text = "Target Strength: ")
l1.grid(row=0, column=0, padx= 5, pady = 5)

ent2 = ttk.Entry(frame, width=15)
ent2.grid(row=1, column=1, padx= 5, pady = 5)
l2 = ttk.Label(frame, text = "Target Diffusion Coefficient: ")
l2.grid(row=1, column=0, padx= 5, pady = 5)

def poyo():
    
    global go
    go = True
    
    rb1.config(state=DISABLED)
    rb2.config(state=DISABLED)
    
    ent1.config(state=DISABLED)
    ent2.config(state=DISABLED)
    
    global F, X, tstr, tdc
    progbar.start(8)
    b_estimate.config(state=DISABLED)
    tstr = float(ent1.get())
    tdc = float(ent2.get())
    F, X = design(tstr, tdc)#(pool.apply_async(design, (50, 1.1))).get()
    progbar.stop()
    
    b_estimate.config(state=NORMAL)
    b_design.config(state=NORMAL)
    
    global mat_select, indx
    mat_select, indx = select(F, X)
    
    mc = mass_of_material(mat_select[0], mat_select[1], mat_select[3], fa, s)
    
    global cncrt
    cncrt = [round(mc[0], 2), round(mc[1],2), round(mat_select[4]), round(mat_select[5]), round(mat_select[6],2)]
    
    Label_text = ('Cement: '+str(cncrt[0])+'\n\n'+
    supc+': '+str(cncrt[1])+'\n\n'
    'Water: '+str(cncrt[2])+'\n\n'
    'Fine Aggregates: '+str(cncrt[3])+'\n\n'
    'Coarse Aggregates: '+str(cncrt[4]))
    
    labr.config(text=Label_text)
    
    rb1.config(state=NORMAL)
    rb2.config(state=NORMAL)
    
    ent1.config(state=NORMAL)
    ent2.config(state=NORMAL)
    
    while (go):
        if len(mc)>0:
            go = False
        pass
    
    return None

def meow():
    global thrd
    thrd = threading.Thread(target=poyo)
    thrd.start()
    return None
    
b_estimate = ttk.Button(frame, text="Estimate", command=lambda:meow())
                       #threading.Thread(target=poyo).start) #Have to stop the thread
b_estimate.grid(row=2, column=0, padx= 1, pady=5)


#ef restart_I_guess():
   #global thrd
   #print("on")
   #thrd.join()
   #return None

'''End of Estimation of Design'''


'''Result area'''

Label_text_init = ('Cement:  '+'0.00'+'\n\n'+
    supc+': 0.00'+'\n\n'
    'Water:  '+'0.00'+'\n\n'
    'Fine Aggregates:  '+'0.00'+'\n\n'
    'Coarse Aggregates:  '+'0.00')

def exprt_csv():
    
    global cncrt
    
    des_res = pd.DataFrame([np.array(cncrt)], columns = ['Cement', supc, 'Water', 'Fine Aggregates', 'Coarse Aggregates'])
    
    save_file = filedialog.asksaveasfilename(filetypes=[('Comma-Separated Values', '*.csv')],
                                             defaultextension = '*.csv',
                                             title='Save As')
    
    des_res.to_csv(save_file)
    
    return None
    

frame_d = ttk.LabelFrame(root, text="Results")
frame_d.grid(row = 4, column = 0, columnspan = 2, padx=10, pady=5)

labr = ttk.Label(frame_d, text = Label_text_init)
labr.grid(row = 0, column = 0, padx=10, pady=20)

b_design = ttk.Button(frame_d, text="Export Design to CSV", command =lambda: exprt_csv(), state=DISABLED) #Have to stop the thread
b_design.grid(row=2, column=1, padx= 1, pady=5)

'''End of Result area'''

root.mainloop()
