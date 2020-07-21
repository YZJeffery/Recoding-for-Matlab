import numpy as np
import math
import json
import sympy
import matplotlib.pyplot as plt
from scipy import optimize as opt
from scipy.integrate import odeint
import rpy2.robjects as robjects

def saveData(filename,data):
    with open(filename, 'w') as file_obj:
        np.savetxt(file_obj,data,delimiter=' ')
        print('Data had been saved into a TXT file: ',filename)

def readData(filename):
    A = np.zeros((160,5),dtype=float)
    with open(filename, 'r') as file_obj:
        lines = file_obj.readlines()
        A_row = 0
        for line in lines:
            wordlist = line.strip('\n').split(' ')
            A[A_row:] = wordlist[0:5]
            A_row += 1
    return A

def exp_func(mtrx):
    phi = np.exp(mtrx)
    return phi

def model(phi,t,dose):
    pred = np.matrix('0.0;0.0;0.0;0.0')
    for count in range(0,4):
        pred[count] = dose/phi[0]*math.exp(-phi[1]*t[count])
    return pred

def final_model(phi,t,dose):
    pred = np.multiply(dose/phi[0],exp_func(np.multiply(-phi[1],t)))
    return pred

def nlmodel(t,phi0,phi1,sigma):
    return 100/(phi0+sigma)*np.exp(-(phi1+sigma)*t)

def nlmodel_beta(inputs,beta1,beta2,beta3,beta4,b1,b2):
    return 100/(beta1+beta3*inputs[0]+b1)*np.exp(-(beta2+beta4*inputs[1]+b2)*inputs[2])

def calculateLE(phi,t,dose):
    y = np.zeros((n*len(times),1))
    j = 0
    for i in range(0,n):
        f = model(phi[i],times,dose)
        y[j] = f[0]+f[0]*reserr*np.random.uniform(0,1)
        y[j+1] = f[1]+f[1]*reserr*np.random.uniform(0,1)
        y[j+2] = f[2]+f[2]*reserr*np.random.uniform(0,1)
        y[j+3] = f[3]+f[3]*reserr*np.random.uniform(0,1)
        j = j+4
    return y

def modelODE(y,t,phi):
    dydt = -phi*y
    return dydt

def calculateODE(phi,t,dose):
    y = np.zeros((n*len(times),1))
    j = 0
    for i in range(0,n):
        new_phi = phi[i]
        y0 = dose/new_phi[0]
        f = odeint(modelODE,y0,tspan,args=(new_phi[1],))
        y[j] = f[0]
        y[j+1] = f[1]
        y[j+2] = f[4]
        y[j+3] = f[8]
        j = j+4
    return y

dose = 100
n = 40
times = np.matrix([0,0.5,2,4]).reshape(4,1)
tspan = np.linspace(0,4,9)
beta0 = np.log([10,0.8])
psi = np.matrix([0.3,0,0,0.1]).reshape(2,2)

#random effects
xb = np.random.multivariate_normal(beta0,psi,n)
wtc = 40+20*np.random.rand(n,1)-50
sex = np.random.binomial(1,0.5,(40,1))
xb[:,0] = xb[:,0] + 0.2*wtc[:,0]
xb[:,1] = xb[:,1] + 0.9*sex[:,0]
phi = exp_func(xb)

reserr = 0.1
groups = np.arange(40)
groups = groups+1
groups1 = groups

Data = readData('PA_Data.txt')
#print(Data.shape)

# Explore the data
subj_id = Data[:,0].reshape(160,1)
time_points = Data[:,1].reshape(160,1)
conc = Data[:,2].reshape(160,1)
body_weight = Data[:,3].reshape(160,1)
subj_sex = Data[:,4].reshape(160,1)

'''
print('body_weight')
print(body_weight)
print('subj_sex')
print(subj_sex)
print('time_points')
print(time_points)
print('conc')
print(conc)
'''


#第一个图，40个个体的浓度-时间曲线
fig_rows = 5
fig_cols = 8
fig1,axes1 = plt.subplots(fig_rows,fig_cols)

count1 = 0
for i in range(0,fig_rows):
    for j in range(0,fig_cols):
        if subj_sex[count1] == 1.0:
            psb = 'red'
        else:
            psb = 'blue'
        t = [time_points[count1,0],time_points[count1+1,0],time_points[count1+2,0],time_points[count1+3,0]]
        c = [conc[count1,0],conc[count1+1,0],conc[count1+2,0],conc[count1+3,0]]
        axes1[i,j].set(xlim = (0,4), ylim = (0.00001,100000))
        axes1[i,j].semilogy(t,c,label='conc',color=psb)
        count1 = count1 + 4
#plt.show()

#第二个图，根据性别将浓度-时间曲线分为两组
fig2,axes2 = plt.subplots(1,2)
axes2[0].set(xlim = (0,4), ylim = (0.00001,10000))
axes2[1].set(xlim = (0,4), ylim = (0.00001,10000))
count2 = 0
for i in range(0,n):
    if subj_sex[count2] == 1.0:
        psb = 'red'
    else:
        psb = 'blue'
    t = [time_points[count2,0],time_points[count2+1,0],time_points[count2+2,0],time_points[count2+3,0]]
    c = [conc[count2,0],conc[count2+1,0],conc[count2+2,0],conc[count2+3,0]]
    axes2[int(subj_sex[count2])].semilogy(t,c,label='conc',color=psb)
    count2 = count2 + 4
#plt.show()

#第三个图，浓度-年龄关系图
plt.figure()
conc_wt_sex = np.zeros((40,3))
count3 = 0

for i in range(0,n):
    part_conc = np.matrix([conc[count3,0],conc[count3+1,0],conc[count3+2,0],conc[count3+3,0]]).reshape(4,1)
    maxconc = part_conc.max()
    indx = np.argmax(part_conc[:,0])
    part_wt = body_weight[count3+indx,0]
    part_sex = subj_sex[count3+indx,0]
    
    count3 = count3 + 4
    conc_wt_sex[i,0] = maxconc
    conc_wt_sex[i,1] = part_wt+50
    conc_wt_sex[i,2] = part_sex

for i in range(0,n):
    if conc_wt_sex[i,2] == 1.0:
        psb = 'red'
        mk = 'o'
    else:
        psb = 'blue'
        mk = '^'
    fig3 = plt.scatter(conc_wt_sex[i,1],conc_wt_sex[i,0],s=None,c=psb,marker=mk)
plt.show()

#curve_fit
#第四个图，用curve_fit来计算beta等值


#此处出现重大错误，用curve_fit拟合推导的不应该是phi的值
#而应该是beta的值
#这样才能进一步用beta、covariates和b来计算phi
#有了phi之后，再用phi去计算y的值
'''
要做：
将原来模型中的phi拆开，用beta，covaiates和b来表示，
再将这个展开后的大式子作为模型来fit，如此可得到beta和b的值，
以便进一步计算matlab中的d、qq1，qq2和qq。
注意控制参数范围。
'''

plt.figure()

#x0 = time_points.flatten()
x0 = [body_weight.flatten(),subj_sex.flatten(),time_points.flatten()]
y0 = conc.flatten()

plt.scatter(time_points,conc,25,'red')

param_bounds = ([-5,-5,-5,-5,0,0],[5,5,5,5,1,1])
beta1_pred = []
beta2_pred = []
beta3_pred = []
beta4_pred = []
b1_pred = []
b2_pred = []
for i in range(0,n):
    beta1,beta2,beta3,beta4,b1,b2 = opt.curve_fit(nlmodel_beta,x0,y0,bounds=param_bounds)[0]
    for j in range(0,4):
        beta1_pred.append(beta1)
        beta2_pred.append(beta2)
        beta3_pred.append(beta3)
        beta4_pred.append(beta4)
        b1_pred.append(b1)
        b2_pred.append(b2)
#print(beta1_pred,beta2_pred,beta3_pred,beta4_pred,b1_pred,b2_pred)
para_pred = [np.array(beta1_pred).flatten(),np.array(beta2_pred).flatten(),np.array(beta3_pred).flatten(),np.array(beta4_pred).flatten(),np.array(b1_pred).flatten(),np.array(b2_pred).flatten()]

d = [para_pred[0],para_pred[1]]

########################
#直接不用para_pred[4]和[5],改用random随机两个数组跟d相加
########################
b1_random = np.random.rand(160,1)
b2_random = np.random.rand(160,1)
b1_random = b1_random.flatten()
b2_random = b2_random.flatten()

qq1 = [(para_pred[0] + b1_random).flatten(),(para_pred[1] + b2_random).flatten()]
qq2 = [np.multiply(para_pred[2],body_weight.flatten()),np.multiply(para_pred[3],subj_sex.flatten())]

qq = qq1 + qq2

pkpar = exp_func(qq)
solu = exp_func(np.array(d).flatten()+np.array(qq2).flatten())

ppop = final_model(solu,time_points.flatten(),100)
pind = final_model(pkpar,time_points.flatten(),100)
obsv = conc

#print(solu)

plt.figure(5)
plt.loglog(ppop,obsv,'r*')
plt.figure(6)
plt.loglog(pind,obsv,'g*')
plt.show()

'''
print(phi0,phi1,sigma)
x3 = np.arange(0,4,0.01)
y3 = 100.0/(phi0+sigma)*np.exp(-(phi1+sigma)*x3)
y4 = 100.0/phi0*np.exp(-phi1*x3)
plt.ylim(0.0001,1000)
plt.semilogy(x3,y3,'purple')
plt.semilogy(x3,y4,'orange')
plt.legend(['individual','population'])
plt.title('test')
plt.xlabel('time')
plt.ylabel('concentration')

plt.show()


接下来的工作：
1. use population data to generate the 'population line'.
2. use ndividual data (4 observations related to 4 timepoints) for each subject to geneerate 'individual lines'.
3. put the 'population line' with each 'indiidual line' to reproduce Listing-5's result.

4. read 3 papers form Zhao, write down the first version for the report befor 6 January.
'''

aa = sympy.Symbol('aa')
fx = 5*aa+20-100
print(sympy.solve([fx],[aa]))


aa1 = sympy.Symbol('aa1')
aa2 = sympy.Symbol('aa2')
aa3 = sympy.Symbol('aa3')
aa4 = sympy.Symbol('aa4')
bb1 = sympy.Symbol('bb1')
bb2 = sympy.Symbol('bb2')
fx1 = 100/(aa1+aa3*5.4635+bb1)*math.e**(-(aa2+aa4*0+bb2)*0) - 1.9730
fx2 = 100/(aa1+aa3*5.4635+bb1)*math.e**(-(aa2+aa4*0+bb2)*0.5) - 1.4813
fx3 = 100/(aa1+aa3*5.4635+bb1)*math.e**(-(aa2+aa4*0+bb2)*2) - 0.6602
fx4 = 100/(aa1+aa3*5.4635+bb1)*math.e**(-(aa2+aa4*0+bb2)*4) - 0.2396
fx5 = 100/(aa1+aa3*9.3994+bb1)*math.e**(-(aa2+aa4*0+bb2)*0) - 1.2066
fx6 = 100/(aa1+aa3*9.3994+bb1)*math.e**(-(aa2+aa4*0+bb2)*0.5) - 0.8270
print(sympy.solve([fx1,fx2,fx3,fx4,fx5,fx6],[aa1,aa2,aa3,aa4,bb1,bb2]))


