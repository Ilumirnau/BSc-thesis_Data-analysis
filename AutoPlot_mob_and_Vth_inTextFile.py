import matplotlib.pyplot as plt
import numpy as np
#print(__file__)

#regression intercept, slope, r_coef and y_pred for x=V_g and y=I_ds
def Regression_Regimes(V_g, I_ds):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    x = np.array(V_g).reshape((-1, 1))
    y = np.array(I_ds)
    model = LinearRegression().fit(x, y)
    r_coef = float(model.score(x, y))
    intercept = float(model.intercept_)
    slope = float(model.coef_)
    y_pred = model.predict(x)
    #plt.plot(x,y_pred, color='coral', lw=2, linestyle='dashed', zorder=1)
    #plt.scatter(x,y, color= 'teal', lw=0.3, zorder=3)
    return (intercept, slope, r_coef, y_pred)  #delete tuple if it does not work

#keep trying regressions with different data and return the best regression
def Regression_optimization_Regimes(V_g, I_ds, addPlot = False):
    minimum_data = int(0.15 * len(V_g))
    maximum_data = int(0.4 * len(V_g))
    amount_of_data = 0
    optimal_V_g = 0
    optimal_I_ds= 0
    intercept = 0
    slope = 0
    r = 0
    y_pred = 0
    amount_of_data_2 = 0
    optimal_V_g_2 = 0
    optimal_I_ds_2 = 0
    intercept_2 = 0
    slope_2 = 0
    y_pred_2 = 0
    r_2 = 0
    for i in range(minimum_data, maximum_data): #rang de nums que agafo
        for j in range(int(len(V_g)/2)-i): #check, on començo a agafar números, shift
            data = Regression_Regimes(V_g[0+j:i+j+1], I_ds[0+j:i+j+1])
            new_r = data[2] #check
            if new_r > r:
                amount_of_data = i
                r = new_r
                optimal_V_g = V_g[0+j:i+j+1]
                optimal_I_ds= I_ds[0+j:i+j+1]
                intercept = data[0]
                slope = data[1]
                y_pred = data[3]
            data2 = Regression_Regimes(V_g[int(len(V_g)/2)+j:int(len(V_g)/2) + i+j+1], I_ds[int(len(V_g)/2)+j:int(len(V_g)/2) + i+j+1])
            new_r_2 = data2[2]
            if new_r_2 > r_2:
                amount_of_data_2 = i
                r_2 = new_r_2
                optimal_V_g_2 = V_g[int(len(V_g)/2) + j : int(len(V_g)/2) + i + j +1]
                optimal_I_ds_2= I_ds[int(len(V_g)/2) + j : int(len(V_g)/2) + i +j +1]
                intercept_2 = data2[0]
                slope_2 = data2[1]
                y_pred_2 = data2[3]
    print('First half: '+str(amount_of_data)+'/'+str(int(len(V_g)/2)), '\n R=' + str(r))
    print('Second half: '+str(amount_of_data_2)+'/'+str(int(len(V_g)/2)), '\n R=' + str(r_2))
    if addPlot:
        if r>=r_2:
            plt.plot(optimal_V_g,y_pred, color='coral', lw=2, linestyle=(0, (5, 4)), zorder=1)
            plt.scatter(optimal_V_g,optimal_I_ds, color= 'teal', lw=0.3, zorder=3)
        else: 
            plt.plot(optimal_V_g_2,y_pred_2, color='coral', lw=2, linestyle=(0, (5, 4)), zorder=1)
            plt.scatter(optimal_V_g_2,optimal_I_ds_2, color= 'teal', lw=0.3, zorder=3)
    if r>=r_2: return (intercept, slope, r) #check tuples created
    else: return (intercept_2, slope_2, r_2)

#returns touple (v_th, mobility) for linear regime
def LinearRegime_params(V_g, I_ds, L, W, V_ds):
    data = Regression_optimization_Regimes(V_g, I_ds, addPlot = False)
    V_th = - data[0]/data[1]
    mobility = data[1]/V_ds * L/W * 1/0.00000001726
    print('Linear_Regime_params calculated')
    return (V_th, mobility)

#returns touple (v_th, mobility) for saturation regime
def SaturationRegime_params(V_g, I_ds, L, W):
    import numpy as np
    sqrtI_ds = [np.sqrt(i) for i in I_ds]
    data = Regression_optimization_Regimes(V_g, sqrtI_ds, addPlot = False)
    V_th = - data[0]/data[1]
    mobility = data[1]**2 * 2*L/W * 1/0.00000001726
    print('Saturation_Regime_params calculated')
    return (V_th, mobility)


#function to plot output data, it takes 2 params:
# path is the adress, the location of the file in the system entered as a string
# file_name is the name of the text file to plot including it's extension, probably .txt, also entered as a string

def output_plot(path, file_name):
    I_ds = []
    V_ds = []
    V_g = []
    file = open(path+file_name)
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
            row_count += 1
        else: 
            V_ds.append(float(row[V_d_index]) - float(row[V_s_index]))
            #row_count += 1
            if V_ds[-1]>0: I_ds.append(-abs(float(row[I_d_index])))
            else: I_ds.append(abs(float(row[I_d_index])))
            V_g.append(float(row[V_g_index]))
    V_g_values = []
    contadors = [] 
    contadors.append(0)
    contador=0
    for i in V_g:
        if i not in V_g_values and contador==0: 
            V_g_values.append(i)
            contador = 1
        elif i not in V_g_values:
            V_g_values.append(i)
            contadors.append(contador)
        else:
            contador+=1  
    contadors.append(contador)
    for i in range(len(V_g_values)):
        plt.plot(V_ds[contadors[i]: contadors[i+1]], I_ds[contadors[i]: contadors[i+1]], label=r'V$_G$' + f'={int(V_g_values[i])} V')
    plt.xlabel(r'$V_{DS}$ (V)')
    plt.ylabel(r'|$I_{DS}$| (A)')
    plt.tick_params(axis='both', which='both', direction='in')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset = False, useMathText=True)
    plt.legend()
    plt.xlim(max(V_ds),min(V_ds))
    plt.savefig('Plot_'+file_name[:-4] + '.png', dpi=500) #change to plt.savefig(file_name[:-4] + '.jpg', dpi=1000) to make it an HD image file
    plt.close()        

#function to plot transfer data, it takes 3 params:
# path is the adress, the location of the file in the system entered as a string
# file_name is the name of the text file to plot including it's extension, probably .txt, also entered as a string
# style is set to linear so it's not necessary to specify, only if 'log' is entered the plot will be logarithic instead of linear
#returns a list with the different touples of V_th and mobility for linear and another one for saturation
def transfer_plot(path, file_name, style='linear', **calculate_params):
    I_ds = []
    V_ds = []
    V_g = []
    file = open(path + file_name) #Output Characteristics p-type [ph-btbtc10,pristine,150um,before annealing,4(1) ; 21_07_06 3_40_40 PM]
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
            I_ds.append(float(row[absID_index]))
            if row_count == 2:
                V_ds_lin = float(row[V_d_index]) - float(row[V_s_index])
            if row_count == len(content) - 1:
                V_ds_sat = float(row[V_d_index]) - float(row[V_s_index])
    V_ds_values = []
    contadors = [] 
    contadors.append(0)
    contador=0
    for i in V_ds:
        if i not in V_ds_values and contador==0: 
            V_ds_values.append(i)
            contador = 1
        elif i not in V_ds_values:
            V_ds_values.append(i)
            contadors.append(contador)
        else:
            contador+=1  
    contadors.append(contador)
    if calculate_params['params'] == True:
        linear_r_params = 0
        saturation_r_params = 0
        for i in range(len(V_ds_values)):
            #uncomment to specify linear and saturation voltage values
            '''
            if V_ds_values[i] == calculate_params['Linear_V_ds']:
                 linear_r_params.append(LinearRegime_params(V_g[contadors[i]: contadors[i+1]], I_ds[contadors[i]: contadors[i+1]], calculate_params['L'], calculate_params['W'], V_ds_values[i]))
                 plt.close()
            elif V_ds_values[i] == calculate_params['Saturation_V_ds']:
                saturation_r_params.append(SaturationRegime_params(V_g[contadors[i]: contadors[i+1]], I_ds[contadors[i]: contadors[i+1]], calculate_params['L'], calculate_params['W']))
                plt.close()
            '''
            #if only 2 V_ds are used and the first is linear and the second is saturated
            if V_ds_values[i] == V_ds_lin:
                 linear_r_params = LinearRegime_params(V_g[contadors[i]: contadors[i+1]], I_ds[contadors[i]: contadors[i+1]], calculate_params['L'], calculate_params['W'], V_ds_values[i])
                 plt.close()
            elif V_ds_values[i] == V_ds_sat:
                saturation_r_params = SaturationRegime_params(V_g[contadors[i]: contadors[i+1]], I_ds[contadors[i]: contadors[i+1]], calculate_params['L'], calculate_params['W'])
                plt.close()
    plt.close()
    for i in range(len(V_ds_values)):
        plt.plot(V_g[contadors[i]: contadors[i+1]], I_ds[contadors[i]: contadors[i+1]], label=r'V$_{DS}$'+f'={int(V_ds_values[i])} V')
    plt.xlabel(r'V$_{G}$ (V)')
    plt.ylabel(r'|I$_{D}$| (A)')
    plt.tick_params(axis='both', which='both', direction='in')
    #plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset = False, useMathText=True)
    plt.legend()
    plt.xlim(max(V_g),min(V_g))
    if style=='log':
        plt.yscale('log')
    if style=='linear':
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useOffset = False, useMathText=True)
    plt.savefig('Plot_'+file_name[:-4] +' '+ style + '.png', dpi=500)
    plt.show()
    plt.close()
    print(file_name, str(style), 'finished')
    if calculate_params['params'] == True: return linear_r_params, saturation_r_params
    

import os
path = os.getcwd()+'/'
print(path)

files = os.listdir(path)
#print(files)
total_params_lin = []
total_params_sat = []
for i in files:
    if i[0:6]=='Output' and i[-4:]=='.txt':
        output_plot(path, i)
    elif i[0:8]=='Transfer' and i[-4:]=='.txt':
        t_lin, t_sat = transfer_plot(path, i, 'linear', params = True, L = 1, W = 100, Saturation_V_ds = -20.0, Linear_V_ds = -2.0)
        total_params_lin.append(t_lin)
        total_params_sat.append(t_sat)
        tl = transfer_plot(path, i, 'log', params = False)
print(total_params_lin)
print(total_params_sat)
print('Linear Regimes params:\n', 'Average V_th=', np.mean([i[0] for i in total_params_lin]),'\n', 'St. dev. (V_th)=', np.std([i[0] for i in total_params_lin]),
      '\n', 'Average mobility=', np.mean([i[1] for i in total_params_lin]), '\n', 'St. dev. (mobility)=', np.std([i[1] for i in total_params_lin]))
print('Saturation Regimes params:\n', 'Average V_th=', np.mean([i[0] for i in total_params_sat]),'\n', 'St. dev. (V_th)=', np.std([i[0] for i in total_params_sat]),
      '\n', 'Average mobility=', np.mean([i[1] for i in total_params_sat]), '\n', 'St. dev. (mobility)=', np.std([i[1] for i in total_params_sat]))
path_names = path.split('\\')
data_analysis_file_name = path_names[-1][:-1]
with open('analysis '+data_analysis_file_name+'.txt', 'w') as f:
    f.write('Linear Regimes params:\n Average V_th='+str(np.mean([i[0] for i in total_params_lin]))+'\n St. dev. (V_th)='+ str(np.std([i[0] for i in total_params_lin]))+
      '\n Average mobility=' + str(np.mean([i[1] for i in total_params_lin])) + '\n St. dev. (mobility)=' + str(np.std([i[1] for i in total_params_lin])))
    f.write(5*'\n')
    f.write('Saturation Regimes params:\n Average V_th='+str(np.mean([i[0] for i in total_params_sat]))+'\n St. dev. (V_th)='+ str(np.std([i[0] for i in total_params_sat]))+
      '\n Average mobility=' + str(np.mean([i[1] for i in total_params_sat])) + '\n St. dev. (mobility)=' + str(np.std([i[1] for i in total_params_sat])))
    

