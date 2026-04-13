import torch
import numpy as np
import matplotlib.pyplot as plt

x_data=[1.0,2.0,3.0]
y_data=[2.0,4.0,6.0]
def forward(x):
    return x*w

def Loss(x,y):
    y_hat=forward(x)
    return (y_hat-y)**2

w_list=[]
mes_list=[]
for w in np.arange(0.0,4.0,0.1):
    l_sum=0
    for x_val,y_val in zip(x_data,y_data):
        y_hat_val=forward(x_val)
        mse_val=Loss(x_val,y_val)
        l_sum+=mse_val
        print('\t',x_val,'\t',y_val,'\t',y_hat_val)
    w_list.append(w)
    mes_list.append(l_sum/len(x_data))
plt.plot(w_list,mes_list)
plt.ylabel('MSE')
plt.xlabel('w')
plt.grid(True)
plt.show()

min_loss_idx=np.argmin(mes_list)
optimal_w=w_list[min_loss_idx]
print('optimal w=',optimal_w)

