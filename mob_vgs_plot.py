def SaturationRegime_params(V_g, I_ds, L, W):
    import numpy as np
    sqrtI_ds = [np.sqrt(i) for i in I_ds]
    slope = (sqrtI_ds[1]-sqrtI_ds[0])/(V_g[1]-V_g[0])
    mobility = slope**2 * 2*L/W * 1/0.00000001726
    avg_Vg = np.mean(V_g)
    #print('Saturation_Regime_params calculated')
    return avg_Vg, mobility


def mob_plot_sat(path, file_name, plot_label, **calculate_params):
    import matplotlib.pyplot as plt
    import numpy as np
    Vg_x = [] #to save the x points of Vg to plot mobility
    mob = []
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
            I_ds.append(float(row[absID_index]))
            if row_count == 2:
                V_ds_lin = float(row[V_d_index]) - float(row[V_s_index])
            if row_count == len(content) - 1:
                V_ds_sat = float(row[V_d_index]) - float(row[V_s_index])
    for i in range(int(3*len(V_g)/4), len(V_g)-4):
        Vg, mobility = SaturationRegime_params((V_g[i], V_g[i+2]), (np.mean([I_ds[i+j-2] for j in range(5)]), np.mean([I_ds[i+j] for j in range(5)])), calculate_params['L'], calculate_params['W'])
        Vg_x.append(Vg)
        mob.append(mobility)
    plt.plot(Vg_x, mob, lw=1, label=plot_label)
    plt.xlabel(r'V$_g$ (V)')
    plt.ylabel(r'Mobility (cm$^2$/(V$\cdot$s))')
    plt.xlim(max(V_g),min(V_g))
    #plt.legend()
    plt.tick_params(axis='both', which='both', direction='in')

import os
import matplotlib.pyplot as plt
path = os.getcwd()+'/'
print(path)

files_p = os.listdir(path)
files=[]
for i in range(len(files_p)):
    if files_p[i].startswith('c') and files_p[i][1]!='6':
        files.append(files_p[i])
path2 = [path+i+'/' for i in files]

length_condition = '150'
direction_condition = 'l'
for i in path2:
    files_c = os.listdir(i)
    for j in range(len(files_c)):
        if not '.' in files_c[j]:
            files_sp = os.listdir(i+'/'+files_c[j])
            for k in range(len(files_sp)):
                if not'.' in files_sp[k]:
                    name = files[path2.index(i)]+' '+files_c[j]+' '+files_sp[k]
                    words = name.split()
                    alkyl = words[1]
                    speed = words[2][:-3]
                    length = words[3][:-2]
                    direction = words[4]
                    if length==length_condition and direction==direction_condition:
                        print(alkyl + ' ' + speed+'mm/s ' + length+ 'um ' + direction)
                        txt_files = os.listdir(i+'/'+files_c[j]+'/'+files_sp[k])
                        for l in txt_files:s
                            if l[-4:]=='.txt' and l[:8]!='analysis' and l[:8]=='Transfer':
                                mob_plot_sat(i+'/'+files_c[j]+'/'+files_sp[k]+'/', l, length+ r'$\mu$m ' + direction, L=1, W=100)
            plt.title('Ph-BTBT-'+alkyl+' '+ speed+'mm/s '+length_condition+r'$\mu$m '+direction_condition)
            plt.savefig(alkyl+' '+ speed+'mms'+'.jpg', dpi=600)
            plt.close()
                    
