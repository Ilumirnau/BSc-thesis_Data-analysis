# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 15:22:05 2022

@author: arnau
"""
#only saturation regimes will be studied

#regression intercept, slope and their uncertainties, r_coef and y_pred for x=V_g and y=I_ds
def Regression_Regimes(V_g, I_ds):
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(V_g).reshape((-1, 1))
    y = np.array(I_ds)
    model = LinearRegression().fit(x, y)
    r_coef = float(model.score(x, y))
    intercept = float(model.intercept_)
    
    slope = float(model.coef_)
    
    S_xx=np.std(x, ddof = 1)**2*(len(x)-1)
    S_yy=np.std(y, ddof = 1)**2*(len(y)-1)
    S_x_y=np.sqrt((S_yy-(S_xx*float(model.coef_)**2))/(len(x)-2))
    
    u_slope = S_x_y/np.sqrt(S_xx)
    
    S_x_y=np.sqrt((S_yy-(S_xx*float(model.coef_)**2))/(len(x)-2))
    x_sq=[]
    for i in x:
      x_sq.append(i**2)
      
    u_intercept = S_x_y * np.sqrt((sum(x_sq))/(len(x)*(len(x)-1)*np.std(x)**2))
    
    y_pred = model.predict(x)
    #plt.plot(x,y_pred, color='coral', lw=2, linestyle='dashed', zorder=1)
    #plt.scatter(x,y, color= 'teal', lw=0.3, zorder=3)
    return (intercept, abs(u_intercept), slope, abs(u_slope), r_coef, y_pred)  #delete tuple if it does not work

#threshold votage function
def Vth(V_g, I_ds):
    import numpy as np
    sqrt_Ids = [np.sqrt(i) for i in I_ds]
    regr_params = Regression_Regimes(V_g, sqrt_Ids)
    vth = -regr_params[0]/regr_params[2]
    u_vth = np.sqrt((-regr_params[1]/regr_params[2])**2 + (regr_params[0]*regr_params[3]/regr_params[2]**2)**2)
    return vth, u_vth

#on/off ratio function, uncertainty I_ds = 1E-12
def OnOffRatio(on, off):
    import numpy as np
    ratio = on/off
    u_ratio = 1E-10/off * np.sqrt(1+ratio)
    return ratio, u_ratio

#subthreshold swing in V/dec
def SS(V_g, I_ds):
    import numpy as np
    logI_ds = [np.log10(i) for i in I_ds]
    regr_params = Regression_Regimes(logI_ds, V_g) #opposite than usual since its ^{-1}
    ss = regr_params[2]
    u_ss = regr_params[3]
    return ss, u_ss

#interstitial traps at 298K for C = 0.00000001726
def N(V_g, I_ds):
    import scipy.constants as ct
    import numpy as np
    s, u_ss = SS(V_g, I_ds)
    n = 0.00000001726/ct.e**2 * ((ct.e * abs(s))/(ct.k * 298 * np.log(10))-1)
    u_n = 0.00000001726/(ct.e * ct.k * 298 * np.log(10)) * u_ss
    return n, u_n

#round result x to sig significant figures
def round_sig(x, sig=2):
    from math import floor,log10
    return round(x, sig-int(floor(log10(abs(x))))-1)

#read data, analyze it and return the desired parameters for 1 file
def ElectChara(path, file_name):
    import numpy as np
    I_ds = []
    V_ds = []
    V_g = [] #data extracted from txt file
    file = open(path + file_name)
    content = file.readlines()
    row_count = 0
    for x in content:
        row = x.split()
        if row_count==0: 
            I_d_index = row.index('ID')
            I_s_index = row.index('IS')
            V_d_index = row.index('VD')
            V_s_index = row.index('VS')
            V_g_index = row.index('VG')
            absID_index = row.index('AbsID')
            row_count += 1
        else: 
            row_count += 1
            V_ds.append(float(row[V_d_index]) - float(row[V_s_index]))
            V_g.append(float(row[V_g_index]))
            I_ds.append(abs(float(row[I_d_index])))
            if row_count == 2:
                V_ds_lin = float(row[V_d_index]) - float(row[V_s_index])
            if row_count == len(content) - 1:
                V_ds_sat = float(row[V_d_index]) - float(row[V_s_index])
    sat_index_i = V_ds.index(V_ds_sat) #this is the first indexs of the vectors with saturation regime data
    sat_index_f = len(I_ds)
    I_ds_sat1 = I_ds[sat_index_i:(sat_index_i+int((sat_index_f-sat_index_i)/2))]
    I_ds_sat2 = I_ds[int(sat_index_i+(sat_index_f-sat_index_i)/2):]
    V_g_sat1 = V_g[sat_index_i:(sat_index_i+int((sat_index_f-sat_index_i)/2))]
    V_g_sat2 = V_g[int(sat_index_i+(sat_index_f-sat_index_i)/2):]
    
    #vth calculations
    try: v_th_index_i = V_g_sat1.index(0)
    except: v_th_index_i = V_g_sat1.index(-0.05)
    v_th_index_f = V_g_sat1.index(-5) #doesnt matter to find it in sat1 or 2, should be the same
    vth1, u_vth1 = Vth(V_g_sat1[v_th_index_i:v_th_index_f], I_ds_sat1[v_th_index_i:v_th_index_f])
    vth2, u_vth2 = Vth(V_g_sat2[v_th_index_i:v_th_index_f], I_ds_sat2[v_th_index_i:v_th_index_f])
    
    #on/off ratio calculations
    off_index_f = V_g_sat1.index(4)
    try: on_index = V_g_sat1.index(-10)
    except: on_index = V_g_sat1.index(-9.95)
    on1 = I_ds_sat1[on_index]
    off1 = np.mean(I_ds_sat1[:off_index_f])

    ratio_onoff1, u_onoff1 = OnOffRatio(on1, off1)
    
    on2 = I_ds_sat2[on_index]
    off2 = np.mean(I_ds_sat2[:off_index_f])
    
    ratio_onoff2, u_onoff2 = OnOffRatio(on2, off2)    
    
    #ss calculations, reuse data from vth to give the same results
    ss1, u_ss1 = SS(V_g_sat1[v_th_index_i:v_th_index_f], I_ds_sat1[v_th_index_i:v_th_index_f])
    ss2, u_ss2 = SS(V_g_sat2[v_th_index_i:v_th_index_f], I_ds_sat2[v_th_index_i:v_th_index_f])
    
    #traps N
    n1, u_n1 = N(V_g_sat1[v_th_index_i:v_th_index_f], I_ds_sat1[v_th_index_i:v_th_index_f])
    n2, u_n2 = N(V_g_sat2[v_th_index_i:v_th_index_f], I_ds_sat2[v_th_index_i:v_th_index_f])
    
    #choose 1 or 2 for the one with min uncertainties
    votes1 = 0
    votes2 = 0
    if u_vth1 < u_vth2:
        votes1 += 1
    else: votes2 *= 1

    if u_onoff1 < u_onoff2:
        votes1 += 1
    else: votes2 *= 1

    if u_ss1 < u_ss2:
        votes1 += 1
    else: votes2 *= 1
    
    if u_n1 < u_n2:
        votes1 += 1
    else: votes2 *= 1
    
    if votes1>= votes2:
        return (vth1, u_vth1, ratio_onoff1, u_onoff1, ss1, u_ss1, n1, u_n1)
    else: return (vth2, u_vth2, ratio_onoff2, u_onoff2, ss2, u_ss2, n2, u_n2)
    
import os
import numpy as np
path = os.getcwd()+'/'
print(path)

files_p = os.listdir(path)
files=[]
for i in range(len(files_p)):
    if files_p[i].startswith('c') and '.' not in files_p[i]:
        files.append(files_p[i])
path2 = [path+i+'/' for i in files]
#pth2 = [path/c6/, etc]
data_total = [['Cn', 'Speed', 'L', 'Direction', 'Vth(V)', 'u_Vth(V)', 'On/off', 'u_on/off', 'SS(V/dec)', 'u_SS(V/dec)', 'N', 'u_N']]
for i in path2:
    files_c = os.listdir(i) #list wth speed folder and others
    for j in range(len(files_c)):
        if not '.' in files_c[j]:
            files_sp = os.listdir(i+'/'+files_c[j]) #list files inside speed folder: directions and lengths and others
            for k in range(len(files_sp)):
                if not'.' in files_sp[k]: #folders with directions and lengths only
                    name = files[path2.index(i)]+' '+files_c[j]+' '+files_sp[k]
                    words = name.split()
                    alkyl = words[1]
                    speed = words[2][:-3]
                    length = words[3][:-2]
                    direction = words[4]
                    
                    print(alkyl + ' ' + speed+'mm/s ' + length+ 'um ' + direction)
                    txt_files = os.listdir(i+'/'+files_c[j]+'/'+files_sp[k])
                    data = []
                    v_ths, u_vths = [], []
                    onoff_ratios, u_ratios = [], []
                    sss, u_sss = [], []
                    ns, u_ns = [], []
                    for l in txt_files:
                        if l[-4:]=='.txt' and l[:8]!='analysis' and l[:8]=='Transfer':
                            data.append(ElectChara(i+'/'+files_c[j]+'/'+files_sp[k]+'/', l))
                    for l in data:
                        v_ths.append(l[0])
                        u_vths.append(l[1])
                        onoff_ratios.append(l[2])
                        u_ratios.append(l[3])
                        sss.append(l[4])
                        u_sss.append(l[5])
                        ns.append(l[6])
                        u_ns.append(l[7])
                    data_means = []
                    data_means.append(np.mean(v_ths))
                    data_means.append(np.sqrt(np.std(u_vths)**2 + np.mean(u_vths)**2))
                    data_means.append(np.mean(onoff_ratios))
                    data_means.append(np.sqrt(np.std(u_ratios)**2 + np.mean(u_ratios)**2))  
                    data_means.append(np.mean(sss))
                    data_means.append(np.sqrt(np.std(u_sss)**2 + np.mean(u_sss)**2))
                    data_means.append(np.mean(ns))
                    data_means.append(np.sqrt(np.std(u_ns)**2 + np.mean(u_ns)**2))
                    total_vector = ['Ph-BTBT-'+alkyl[1:]+' '+speed+' '+length+' '+direction]
                    for l in data_means:
                        total_vector.append(l)
                    data_total.append(total_vector)
with open('analysis electric_vf'+'.txt', 'w') as f:
   for i in range(len(data_total)):
       if i==0:
           for j in data_total[i]:
               f.write(j+' ')
           f.write('\n')
       else:
           for j in data_total[i]:
               f.write(str(j) +' ')
           f.write('\n')
