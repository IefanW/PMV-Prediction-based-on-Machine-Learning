import math
import numpy as np
import pymysql
import os
from decimal import Decimal
import pandas as pd

if(os.path.exists("simulation_data.csv")):
    os.remove("simulation_data.csv")

# Define list to write into csv
M_list = []; W_list = []; Icl_list = []; Ta_list = []; RH_list = []; v_list = []; Tr_list = []; PMV_list = []; PMV_category_list = []


np.random.seed(0)
for i in range(100000):

    # Variable definition
    M = np.random.randint(58,70) # Metabolism
    W = 0 # Work
    Icl = float(Decimal(np.random.uniform(0.3,0.5)).quantize(Decimal("0.000"))) # 服装热阻
    Ta_Celsius = np.random.uniform(27,30) # 空气温度
    Related_humidity = float(np.random.uniform(55,65)/100) # 相对湿度
    v = float(Decimal(np.random.uniform(0.3,0.5)).quantize(Decimal("0.00"))) #风速
    Tr_Celsius = Ta_Celsius # 平均辐射温度

    # Calculate pa
    ps = 0.611 * math.exp(17.269 * Ta_Celsius / (Ta_Celsius+237.3))
    pa = Related_humidity * ps # 计算人体周围空气的水蒸气压力
    M_W = M - W

    # Tolerance definition
    Tolerance = 0.00015

    # Calculate FCl Value
    FCL = 1.05 + 0.1*Icl # 穿衣人体外表面积与裸体表面积之比，原公式中Icl的单位为clo，
    if Icl < 0.5:        # 而1clo = 0.155km2/w,转换后得此方程
         FCL = 1.0 + 0.2*Icl

    # First guess for surface temperature
    Ta_Kelvin = Ta_Celsius + 273 #convert to Calvin
    Tr_Kelvin = Tr_Celsius + 273
    TCL_Assumed = 0
    XN = TCL_Assumed/100
    XF = XN

    # Compute Surface Temperature of Clothing by Successive Substitution Iterations
    IFCL = Icl * 0.155 * FCL
    P1 = IFCL*Ta_Kelvin
    P2 = IFCL*3.96
    P3 = IFCL*100
    P4 = 35.7 +273- 0.028*M_W + P2*((Tr_Kelvin/100)**4) #1/100**4 offsets 10**-8
    nIterations=0

    while nIterations<150:
        XF = (XF+XN) / 2

    # HC Calculation
        HCF = 12.1 * (v**0.5)
        HCN = 2.38 * abs((100*XF-Ta_Kelvin)**0.25) #100*XF = tcl
        if HCF > HCN:
            HC = HCF
        else:
            HC = HCN
            
        # Iterative computation of TCL
        XN = (P4+P1*HC-P2*(XF**4)) / (100+P3*HC)
        nIterations += 1
        if ((nIterations>1) & (abs(XN-XF)<Tolerance)):
            break

        if nIterations < 150:
            TCL = 100*XN - 273

    # Compute the Predicted Mean Vote (PMV)
            R = 3.96 * FCL * ((XN**4-(Tr_Kelvin/100)**4)) # R
            C = FCL * HC * (TCL-Ta_Celsius) # C
            Head = 0.303 * math.exp(-0.036*M) + 0.028 # Main function
            E_sw = 0.0
            if M_W > 58.15:
                E_sw = 0.42 * (M_W-58.15) #Esw
            Func_p1 = M_W - 3.05*0.001*(5733-6.99*M_W-pa) # M-W-Ed
            Func_p2 = -E_sw - 1.7*0.00001*M*(5867-pa) - 0.0014*M*(34-Ta_Celsius) - R - C # -Esw - Eres - L - R - C
            PMV = Head * (Func_p1+Func_p2) # Result
        else:
            PMV=999

    # Appending data
    M_list.append(float(M)); W_list.append(float(W)); Icl_list.append(Icl); Ta_list.append(float(Ta_Celsius))
    RH_list.append(float(Related_humidity)); v_list.append(v); Tr_list.append(float(Tr_Celsius))
    PMV_list.append(PMV); PMV_category_list.append(round(PMV))

df = pd.DataFrame({'Metabolic_rate': M_list, 'W': W_list, 'Iclo': Icl_list, 'Air_temp': Ta_list,
                   'Related_humidity': RH_list, 'Air_speed': v_list,
                   'Radiation_temp': Tr_list, 'PMV': PMV_list, 'PMV_category': PMV_category_list})
print(df.head())
df.to_csv("simulation_data.csv", index=False)
