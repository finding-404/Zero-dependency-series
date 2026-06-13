import numpy as np, yfinance as yf, pandas as pd, pandas_market_calendars as mcal, matplotlib.pyplot as plt, time as timing
from datetime import *
from zoneinfo import ZoneInfo

# data collection
nyse=mcal.get_calendar('NYSE')
now_ny=datetime.now(ZoneInfo('America/New_York'))
market_open=nyse.schedule(start_date=now_ny.date(), end_date=now_ny.date())

now_chi=datetime.now(ZoneInfo('America/Chicago'))

if not(market_open.empty) and market_open.iloc[0]['market_open'] <= now_ny <= market_open.iloc[0]['market_close']:
        qqqm=yf.download('QQQM', period='max')[:-1]
        ndx_future=yf.download('NQ=F', start=qqqm.index.date[0], end=None)[:-1]
        vix=yf.download('^VIX', start=qqqm.index.date[0], end=None)[:-1]
else:
    if now_ny.weekday()==5 or now_ny.weekday()==6:
        qqqm=yf.download('QQQM', period='max')
        ndx_future=yf.download('NQ=F', start=qqqm.index.date[0], end=None)
        vix=yf.download('^VIX', start=qqqm.index.date[0], end=None)
    elif now_ny.date()==now_chi.date():
        qqqm=yf.download('QQQM', period='max')
        ndx_future=yf.download('NQ=F', start=qqqm.index.date[0], end=None)[:-1]
        vix=yf.download('^VIX', start=qqqm.index.date[0], end=None)[:-1]
    else:
        qqqm=yf.download('QQQM', period='max')
        ndx_future=yf.download('NQ=F', start=qqqm.index.date[0], end=None)
        vix=yf.download('^VIX', start=qqqm.index.date[0], end=None)

print(qqqm)
print(ndx_future)
print(vix)
print()

calander=nyse.schedule(start_date=f'{now_ny.year}-{now_ny.month}-01', end_date=f'{now_ny.year}-{now_ny.month+2}-01').index.date
tomorrow_open_day_index=np.argmax(calander==qqqm.index.date[-1])+1
print(calander[tomorrow_open_day_index])
time_interval=np.atleast_2d(np.abs(qqqm.index[:-1]-qqqm.index[1:])/timedelta(days=1)).T
time_interval=np.concatenate((time_interval, [[(calander[tomorrow_open_day_index]-qqqm.index.date[-1])/timedelta(days=1)]]))
print(time_interval)

merged=qqqm.join(ndx_future, lsuffix='_left', rsuffix='_right').join(vix, lsuffix='_left', rsuffix='_right').to_numpy()
ndx_future=merged[:, 5:10]
vix=merged[:, 10:15]/1000 # vix index normalize

# FFT featuring extration for QQQM Price and NDX Future
qqqm_fft=[]
ndx_fft=[]
for i in range(14, len(qqqm)):
    qqqm_fft.append(np.log(1+np.abs(np.concatenate((np.fft.rfft(qqqm.to_numpy()[i-14:i, 0]), np.fft.rfft(qqqm.to_numpy()[i-14:i, 1]), np.fft.rfft(qqqm.to_numpy()[i-14:i, 2]), np.fft.rfft(qqqm.to_numpy()[i-14:i, 3])))))/100)
    ndx_fft.append(np.log(1+np.abs(np.concatenate((np.fft.rfft(ndx_future[i-14:i, 0]), np.fft.rfft(ndx_future[i-14:i, 1]), np.fft.rfft(ndx_future[i-14:i, 2]), np.fft.rfft(ndx_future[i-14:i, 3])))))/100)

# QQQM price normalization
new_qqqm=merged[:, 0:5]
new_qqqm[1:, 0:4]=np.log(new_qqqm[1:, 0:4]/new_qqqm[:-1, 0:4])
for i in range(20, len(qqqm)):
    new_qqqm[i, 4]=np.log(new_qqqm[i, 4]/np.median(qqqm.to_numpy()[i-20:i, 4]))


def huber_loss(batch_data, batch_answer, h):
    mse_signal=np.abs(batch_answer-batch_data)<=h
    mse_loss=(1/2)*((batch_answer-batch_data)**2)*mse_signal

    mae_signal=~mse_signal
    mae_loss=h*(np.abs(batch_answer-batch_data)-(1/2)*h)*mae_signal

    return np.mean(mse_loss+mae_loss, axis=1, keepdims=True)

def d_huber(batch_data, batch_answer, h):
    mse_signal=np.abs(batch_answer-batch_data)<=h
    d_mse=-(batch_answer-batch_data)*mse_signal

    mae_signal=~mse_signal
    d_mae=-h*np.sign(batch_answer-batch_data)*mae_signal

    return (d_mse+d_mae)/(len(batch_data[0])*len(batch_data))

class MLP:
    def __init__(self, layers, activation, normalization, norm_param, optimizer):
        self.activation_list=activation
        self.norm_list=normalization
        self.norm_param_list=norm_param
        self.params_num=0
        self.params=self.layers(layers, np.float64)
        self.optimizer=optimizer

        if self.optimizer=='Adam':
            self.time=1
            self.mean=np.zeros(self.params_num)
            self.variance=np.zeros(self.params_num)

        elif self.optimizer=='RMSprop':
            self.time=1
            self.variance=np.zeros(self.params_num)

        elif self.optimizer=='AdaGrad':
            self.time=1
            self.gradient_accum=np.zeros(self.params_num)

        elif self.optimizer=='Momentum':
            self.velocity=np.zeros(self.params_num)


    def layers(self, structure, data_type):
        params=[]
        for i in range(0, len(structure)-1):
            if self.norm_list[i]=='LN' and self.norm_param_list[i]:
                parameter=[np.random.normal(0, np.sqrt(2/structure[i]), (structure[i+1], structure[i])).astype(data_type),
                           np.random.normal(0, np.sqrt(2/structure[i+1]), (1, structure[i+1])).astype(data_type),
                           np.random.normal(0, np.sqrt(2/structure[i+1]), (1, structure[i+1])).astype(data_type),
                           np.random.normal(0, np.sqrt(2/structure[i+1]), (1, structure[i+1])).astype(data_type)]
                params.append(parameter)
                self.params_num+=np.size(parameter[0])+np.size(parameter[1])+np.size(parameter[2])+np.size(parameter[3])
            elif self.norm_list[i]=='RMS' and self.norm_param_list[i]:
                parameter=[np.random.normal(0, np.sqrt(2/structure[i]), (structure[i+1], structure[i])).astype(data_type),
                           np.random.normal(0, np.sqrt(2/structure[i+1]), (1, structure[i+1])).astype(data_type),
                           np.random.normal(0, np.sqrt(2/structure[i+1]), (1, structure[i+1])).astype(data_type)]
                params.append(parameter)
                self.params_num+=np.size(parameter[0])+np.size(parameter[1])+np.size(parameter[2])
            else:
                parameter=[np.random.normal(0, np.sqrt(2/structure[i]), (structure[i+1], structure[i])).astype(data_type),
                           np.random.normal(0, np.sqrt(2/structure[i+1]), (1, structure[i+1])).astype(data_type)]
                params.append(parameter)
                self.params_num+=np.size(parameter[0])+np.size(parameter[1])
        return params

    def activation(self, data, act_type):
        if act_type=='ReLU':
            return np.maximum(0, data)
        elif act_type=='LeakyReLU':
            return np.maximum(0.01*data, data)
        elif act_type=='SiLU':
            data=np.clip(data, -700, 700)
            return data*np.where(data >= 0, 1/(1+np.exp(-data)), np.exp(data)/(1+np.exp(data)))
        elif act_type=='Softplus':
            data=np.clip(data, -700, 700)
            return np.log(1+np.exp(-np.abs(data)))+np.maximum(0, data)
        elif act_type=='Sigmoid':
            data=np.clip(data, -700, 700)
            return np.where(data >= 0, 1/(1+np.exp(-data)), np.exp(data)/(1+np.exp(data)))
        elif act_type=='Tanh':
            return np.tanh(data)
        elif act_type=='identity':
            return data
        else:
            return data

    def d_activation(self, data, act_type):
            if act_type=='ReLU':
                return (data>0).astype(np.float64)
            elif act_type=='LeakyReLU':
                return np.clip((data>0.01*data), 0.01, 1, dtype=np.float64)
            elif act_type=='SiLU':
                data=np.clip(data, -700, 700)
                return 1/(1+np.exp(-data))+data*self.activation(data, 'Sigmoid')*(1-self.activation(data, 'Sigmoid'))
            elif act_type=='Softplus':
                data=np.clip(data, -700, 700)
                return 1/(1+np.exp(-data))
            elif act_type=='Sigmoid':
                data=np.clip(data, -700, 700)
                return self.activate(data, 'Sigmoid')*(1-self.activate(data, 'Sigmoid'))
            elif act_type=='Tanh':
                return 1-np.tanh(data)**2 
            elif act_type=='identity':
                return np.ones_like(data)   
            else:
                return np.ones_like(data)  
        
    def normalize(self, data, norm_type):
        if norm_type=='LN':
            mean=np.mean(data, axis=1, keepdims=True)
            var=np.mean((data-mean)**2, axis=1, keepdims=True)
            return (data-mean)/np.sqrt(var+1e-8), mean, np.sqrt(var+1e-8)  
        
        elif norm_type=='RMS':
            mean=np.mean(data**2, axis=1, keepdims=True)
            return data/np.sqrt(mean+1e-8), np.sqrt(mean+1e-8)
        
        elif norm_type=='identity':
            return data, None

    def d_normalize(self, data, loss, norm_type):
        norm_value=self.normalize(data, norm_type)
        if norm_type=='LN':
            return (1/(len(data[0])*norm_value[2]))*(len(data[0])*loss-np.sum(loss, axis=1, keepdims=True)-norm_value[0]*np.sum(loss*norm_value[0], axis=1, keepdims=True))
        elif norm_type=='RMS':
            return (1/norm_value[1])*(loss-(data/(norm_value[1]**2))*(1/len(loss[0]))*np.sum(loss*data, axis=1, keepdims=True))    
        elif norm_type=='identity':
            return loss
    
    def forward(self, data):
        output=[]
        result=data
        for i in range(len(self.params)):
            z=np.dot(result, self.params[i][0].T)+self.params[i][1]
            a=self.activation(z, self.activation_list[i])
            n=self.normalize(a, self.norm_list[i])[0]

            if self.norm_list[i]=='LN' and self.norm_param_list[i]:
                sn=n*self.params[i][2]+self.params[i][3]
                output.append([result, z, a, n, sn])
                result=sn
    
            elif self.norm_list[i]=='RMS' and self.norm_param_list[i]:
                sn=n*self.params[i][2]
                output.append([result, z, a, n, sn])
                result=sn

            else:
                output.append([result, z, a, n])
                result=n
        return output

    def backward(self, data, loss, insert_loss=None):
        memory=self.forward(data)
        gradients=[]
        previous_loss=loss
        for i in reversed(range(0, len(self.params))):
            if self.norm_list[i]=='LN' and self.norm_param_list[i]:
                g_shifter=previous_loss
                g_scalar=g_shifter*memory[i][3]
                g_norm=self.d_normalize(memory[i][2], g_shifter*self.params[i][2], self.norm_list[i])*self.params[i][2]

                gradients.append(np.ravel(np.mean(g_shifter, axis=0)))
                gradients.append(np.ravel(np.mean(g_scalar, axis=0)))
                
            elif self.norm_list[i]=='RMS' and self.norm_param_list[i]:
                g_scalar=previous_loss*memory[i][3]
                g_norm=self.d_normalize(memory[i][2], previous_loss*self.params[i][2], self.norm_list[i])*self.params[i][2]

                gradients.append(np.ravel(np.mean(g_scalar, axis=0)))
            else:
                g_norm=self.d_normalize(memory[i][2], previous_loss, self.norm_list[i])

            g_bias=g_norm*self.d_activation(memory[i][1], self.activation_list[i])
            g_weight=np.dot(g_bias.T, memory[i][0])
            g_input=np.dot(g_bias, self.params[i][0])
            previous_loss=g_input
            if insert_loss!=None:
                previous_loss+=insert_loss[i]

            gradients.append(np.ravel(np.mean(g_bias, axis=0)))
            gradients.append(np.ravel(g_weight/len(data)))

        return np.concatenate((gradients)), previous_loss

    def optimize(self, gradient, lr, beta_1, beta_2, lamb=0,  optimize_layer=None):
        final_gradient=gradient
        if self.optimizer=='Adam':
            self.mean=beta_1*self.mean+(1-beta_1)*gradient
            self.variance=beta_2*self.variance+(1-beta_2)*(gradient**2)
        
            new_mean=self.mean/(1-(beta_1**self.time))
            new_variance=self.variance/(1-(beta_2**self.time))
        
            final_gradient=list(new_mean/(np.sqrt(new_variance)+1e-8))
            self.time+=1

        elif self.optimizer=='RMSprop':
            self.variance=beta_1*self.variance+(1-beta_1)*(gradient**2)
            final_gradient=list(gradient/np.sqrt(self.variance+1e-8))
            self.time+=1

        elif self.optimizer=='AdaGrad':
            self.gradient_accum+=gradient**2
            final_gradient=list(gradient/np.sqrt(self.gradient_accum+1e-8))
            self.time+=1

        elif self.optimizer=='Momentum':
            self.velocity=beta_1*self.velocity+(1-beta_1)*gradient
            final_gradient=list(self.velocity)

        elif self.optimizer=='GD':
            final_gradient=list(gradient)
        else:
            final_gradient=list(gradient)

        for i in reversed(range(len(self.params))):
            if self.norm_list[i]=='LN' and self.norm_param_list[i]:
                self.params[i][3]-=lr*np.array([final_gradient.pop(0) for _ in range(self.params[i][3].size)], dtype=np.float64)*np.array(optimize_layer)[i]
                self.params[i][2]-=lr*np.array([final_gradient.pop(0) for _ in range(self.params[i][2].size)], dtype=np.float64)*np.array(optimize_layer)[i]
            elif self.norm_list[i]=='RMS' and self.norm_param_list[i]:
                self.params[i][2]-=lr*np.array([final_gradient.pop(0) for _ in range(self.params[i][2].size)], dtype=np.float64)*np.array(optimize_layer)[i]
            self.params[i][1]-=lr*np.array([final_gradient.pop(0) for _ in range(self.params[i][1].size)], dtype=np.float64)*np.array(optimize_layer)[i]
            self.params[i][0]-=lr*(np.array([final_gradient.pop(0) for _ in range(self.params[i][0].size)], dtype=np.float64).reshape(np.shape(self.params[i][0]))+lamb*self.params[i][0])*np.array(optimize_layer)[i]

    def clear_optimizer_memory(self):
        if self.optimizer=='Adam':
            self.time=1
            self.mean=np.zeros(self.params_num)
            self.variance=np.zeros(self.params_num)
        elif self.optimizer=='RMSprop':
            self.time=1
            self.variance=np.zeros(self.params_num)
        elif self.optimizer=='AdaGrad':
            self.time=1
            self.gradient_accum=np.zeros(self.params_num)
        elif self.optimizer=='Momentum':
            self.velocity=np.zeros(self.params_num)
            
class AdamNODE:
    def __init__(self, layers, activation, normalization, norm_param, optimizer):
        self.M_MLP=MLP(layers[0], activation[0], normalization[0], norm_param[0], optimizer[0])
        self.V_MLP=MLP(layers[1], activation[1], normalization[1], norm_param[1],  optimizer[1])
        self.Drag=MLP(layers[2], activation[2], normalization[2], norm_param[2], optimizer[2])
        self.Diffusion=MLP(layers[3], activation[3], normalization[3], norm_param[3], optimizer[3])

        self.momentum=[]
        self.variance=[]
        self.force=[]
        self.drag=[]
        self.diffusion=[]
        self.noise=[]
        self.hidden_state=[]

        self.index_1=layers[0][0]-1
        self.index_2=layers[0][0]-1+self.index_1
        self.index_3=layers[0][-1]+self.index_2

        self.gradient=[np.zeros(self.M_MLP.params_num), np.zeros(self.V_MLP.params_num), np.zeros(self.Drag.params_num), np.zeros(self.Diffusion.params_num)]

    def clean_memory(self):
        self.momentum=[]
        self.variance=[]
        self.force=[]
        self.drag=[]
        self.diffusion=[]
        self.noise=[]
        self.hidden_state=[]
    
    def forward(self, data, time, step):
        momentum=data[:, 0:self.index_1]
        variance=data[:, self.index_1 : self.index_2]
        hidden_state=data[:, self.index_2 : self.index_3]
        self.momentum.append(momentum)
        self.variance.append(variance)
        self.hidden_state.append(hidden_state)

        delta_t=time/step
        for i in range(0,step):
            momentum+=(1-0.9)*(-1*self.M_MLP.forward(np.concatenate((hidden_state, delta_t), axis=1))[-1][-1]-momentum)
            variance+=(1-0.99)*((self.V_MLP.forward(np.concatenate((hidden_state, delta_t), axis=1))[-1][-1]**2)-variance)

            force=(-1*momentum/(np.sqrt(variance)+1e-8))
            drag=self.Drag.forward(np.concatenate((hidden_state, data[:, self.index_3:], delta_t), axis=1))[-1][-1]
            diffusion=self.Diffusion.forward(np.concatenate((hidden_state, data[:, self.index_3:], delta_t), axis=1))[-1][-1]
            noise=np.random.normal(0, 1, (len(data), self.index_1))

            hidden_state=((hidden_state+delta_t*force)+(diffusion*np.sqrt(delta_t)*noise))/(1+drag*delta_t)

            self.momentum.append(momentum)
            self.variance.append(variance)
            self.force.append(force)
            self.drag.append(drag)
            self.diffusion.append(diffusion)
            self.noise.append(noise)
            self.hidden_state.append(hidden_state)

        return np.concatenate((momentum, variance, hidden_state), axis=1)

    def backward(self, data, loss, time, step):
        am=loss[:, 0:self.index_1]
        av=loss[:, self.index_1 : self.index_2]
        ah=loss[:, self.index_2 : self.index_3]
        delta_t=time/step
        for i in reversed(range(0,step)):
            da_momentum=ah*delta_t*(1/(1+self.drag[i]*delta_t))*(-1/np.sqrt(self.variance[i]+1e-8))*(1-0.9)*(-1)
            da_variance=ah*delta_t*(1/(1+self.drag[i]*delta_t))*(-1*self.momentum[i])*(-1/(2*(np.sqrt(self.variance[i])+1e-8)))*(1-0.99)*2
            da_drag=ah*(-self.hidden_state[i+1]*(delta_t/(1+self.drag[i]*delta_t)))
            da_diffusion=ah*(1/(1+self.drag[i]*delta_t))*np.sqrt(delta_t)*self.noise[i]
        
            momentum_gradient=self.M_MLP.backward(np.concatenate((self.hidden_state[i], delta_t), axis=1), da_momentum)
            variance_gradient=self.V_MLP.backward(np.concatenate((self.hidden_state[i], delta_t), axis=1), da_variance)
            drag_gradient=self.Drag.backward(np.concatenate((self.hidden_state[i], data[:, self.index_3:], delta_t), axis=1), da_drag)
            diffusion_gradient=self.Diffusion.backward(np.concatenate((self.hidden_state[i], data[:, self.index_3:], delta_t), axis=1), da_diffusion)

            self.gradient[0]+=momentum_gradient[0]
            self.gradient[1]+=variance_gradient[0]
            self.gradient[2]+=drag_gradient[0]
            self.gradient[3]+=diffusion_gradient[0]
        
            am+=ah*delta_t*(1/(1+self.drag[i]*delta_t))*(-1/np.sqrt(self.variance[i]+1e-8))*(1+(1-0.9)*(-1))
            av+=ah*delta_t*(1/(1+self.drag[i]*delta_t))*(-1*self.momentum[i])*(-1/(2*(np.sqrt(self.variance[i])+1e-8)))*(1+(1-0.99)*(-1))
            ah=ah*(1/(1+self.drag[i]*delta_t))+momentum_gradient[-1][:, :self.index_1]+variance_gradient[-1][:, :self.index_1]+drag_gradient[-1][:, :self.index_1]+diffusion_gradient[-1][:, :self.index_1]
        
        self.clean_memory()
        return np.concatenate((am, av, ah), axis=1)

    def optimize(self, lr, beta_1, beta_2, lamb=0, optimize_layer=None):
        self.M_MLP.optimize(self.gradient[0], lr, beta_1, beta_2, lamb, optimize_layer)
        self.V_MLP.optimize(self.gradient[1], lr, beta_1, beta_2, lamb, optimize_layer)
        self.Drag.optimize(self.gradient[2], lr, beta_1, beta_2, lamb, optimize_layer)
        self.Diffusion.optimize(self.gradient[3], lr, beta_1, beta_2, lamb, optimize_layer)
        self.gradient=[np.zeros(self.M_MLP.params_num), np.zeros(self.V_MLP.params_num), np.zeros(self.Drag.params_num), np.zeros(self.Diffusion.params_num)]

    def clear_optimizer_memory(self):
        self.M_MLP.clear_optimizer_memory()
        self.V_MLP.clear_optimizer_memory()
        self.Drag.clear_optimizer_memory()
        self.Diffusion.clear_optimizer_memory()

def candle_layer(data, scale): # make sure that the output is a correct price candle
    Close=data[:, 0]
    High=np.maximum(0, data[:, 0])+scale*(np.log(1+np.exp(-np.abs(data[:, 1])))+np.maximum(0, data[:, 1]))
    Low=np.minimum(0, data[:, 0])-scale*(np.log(1+np.exp(-np.abs(data[:, 2])))+np.maximum(0, data[:, 2]))

    return np.array([Close, High, Low]).T

def d_candle_layer(data, scale, loss):
    d_Close=loss[:, 0]+loss[:, 0]*np.array(data[:, 0]>0, dtype=np.int8)+loss[:, 0]*np.array(data[:, 0]<0, dtype=np.int8)
    d_High=scale*(1/(1+np.exp(-data[:, 1])))*loss[:, 1]
    d_Low=scale*(-1/(1+np.exp(-data[:, 2])))*loss[:, 2]

    return np.array([d_Close, d_High, d_Low]).T


# model structure setting
optimizer='Adam'

overnight_map_in=MLP([5+32+32,45, 120], ['SiLU', 'Softplus'], ['RMS', 'identity'], [False, False], optimizer)
overnight_ode=AdamNODE([[40+1, 50, 40], 
                        [40+1, 50, 40],
                        [40+4+1, 50, 40],
                        [40+4+1, 50, 40]], [['SiLU', 'identity'],['SiLU', 'identity'],['SiLU', 'Softplus'],['SiLU', 'Tanh']], [['LN', 'identity']]*4, [[False, False]]*4, [optimizer]*4)
overnight_map_out=MLP([120,45,1], ['SiLU', 'identity'], ['LN', 'identity'], [True, False], optimizer)

now_map_in=MLP([45+32+32, 45, 90], ['SiLU', 'Softplus'], ['RMS', 'identity'], [False, False], optimizer)
now_ode=AdamNODE([[30+1, 50, 30], 
                  [30+1, 50, 30],
                  [30+4+1, 50, 30],
                  [30+4+1, 50, 30]], [['SiLU', 'identity'],['SiLU', 'identity'],['SiLU', 'Softplus'],['SiLU', 'Tanh']], [['LN', 'identity']]*4, [[False, False]]*4, [optimizer]*4)
now_map_out=MLP([90,45,3], ['SiLU', 'identity'], ['LN', 'identity'], [True, False], optimizer)

# dataset and answer setting
train_qqqm=new_qqqm[20:-1]
train_vix=vix[20:-1, 0:4]
train_qqqm_fft=qqqm_fft[6:-1]
train_ndx_fft=ndx_fft[6:-1]
train_time_interval=time_interval[20:-1]/10

overnight_ans=np.atleast_2d(np.log(qqqm.to_numpy()[21:, 3]/qqqm.to_numpy()[20:-1, 0])).T
now_ans=np.log(np.array([qqqm.to_numpy()[21:-1, 0]/qqqm.to_numpy()[21:-1, 3],
                         qqqm.to_numpy()[21:-1, 1]/qqqm.to_numpy()[21:-1, 3],
                         qqqm.to_numpy()[21:-1, 2]/qqqm.to_numpy()[21:-1, 3]]).T)

# hyperparamater setting
steps=5
learning_rate=0.003
beta_1=0.9
beta_2=0.99
lamb=0.02 # if lamb>0 and optimizer is Adam, it become AdamW
epochs=100
sim_num=1000 # how many situation should the Monte Carlo simulate
k_scale_factor=0.1 # a scalar to limit the price range of prediction

# Training
loss_list=[]
for t in range(0,epochs):
    if (t+1)>=50 and (t+1)%10==0:
        learning_rate*=0.8

    t1=timing.time()
    over_maped_1=overnight_map_in.forward(np.concatenate((train_qqqm[:-1], train_qqqm_fft[:-1], train_ndx_fft[:-1]), axis=1))
    over_int=overnight_ode.forward(np.concatenate((over_maped_1[-1][-1], train_vix[:-1]), axis=1), train_time_interval[:-1], steps)
    over_maped_2=overnight_map_out.forward(over_int+over_maped_1[-1][-1])

    now_maped_1=now_map_in.forward(np.concatenate((over_maped_2[-2][-1], train_qqqm_fft[:-1], train_ndx_fft[:-1]), axis=1))
    now_int=now_ode.forward(np.concatenate((now_maped_1[-1][-1], train_vix[:-1]), axis=1), train_time_interval[:-1], steps)
    now_maped_2_raw=now_map_out.forward(now_int+now_maped_1[-1][-1])
    now_maped_2=candle_layer(now_maped_2_raw[-1][-1], k_scale_factor)

    loss_overnight=huber_loss(over_maped_2[-1][-1], overnight_ans[:-1], 0.005)
    loss_now=huber_loss(now_maped_2, now_ans, 0.005)

    dl_dy=d_huber(now_maped_2, now_ans, 0.005)
    dl_candle=d_candle_layer(now_maped_2_raw[-1][-1], k_scale_factor, dl_dy)
    #print(dl_dy)
    #print()

    g_now_map_out=now_map_out.backward(now_int, dl_candle)
    g_now_int=now_ode.backward(np.concatenate((now_maped_1[-1][-1], train_vix[:-1]), axis=1), g_now_map_out[-1], train_time_interval[:-1], steps)
    g_now_map_in=now_map_in.backward(np.concatenate((over_maped_2[-2][-1], train_qqqm_fft[:-1], train_ndx_fft[:-1]), axis=1), g_now_int+g_now_map_out[-1])

    dl_dy=d_huber(over_maped_2[-1][-1], overnight_ans[:-1], 0.005)
    
    g_over_map_out=overnight_map_out.backward(over_int, dl_dy, [0, g_now_map_in[-1][:, 0:45]])
    g_over_int=overnight_ode.backward(np.concatenate((over_maped_1[-1][-1], train_vix[:-1]), axis=1), g_over_map_out[-1], train_time_interval[:-1], steps)
    g_over_map_in=overnight_map_in.backward(np.concatenate((train_qqqm[:-1], train_qqqm_fft[:-1], train_ndx_fft[:-1]), axis=1), g_over_int+g_over_map_out[-1])

    now_map_out.optimize(g_now_map_out[0], learning_rate, beta_1, beta_2, lamb, [True, True])
    now_ode.optimize(learning_rate, beta_1, beta_2, lamb, [True, True])
    now_map_in.optimize(g_now_map_in[0], learning_rate, beta_1, beta_2, lamb, [True, True])

    overnight_map_out.optimize(g_over_map_out[0], learning_rate, beta_1, beta_2, lamb, [True, False])
    overnight_ode.optimize(learning_rate, beta_1, beta_2, lamb, [True, True])
    overnight_map_in.optimize(g_over_map_in[0], learning_rate, beta_1, beta_2, lamb, [True, True])
    t2=timing.time()

    loss_list.append([np.mean(loss_overnight), np.mean(loss_now)])

    print(f'the {t+1}th training loss...')
    print(pd.DataFrame(np.concatenate((loss_overnight, loss_now), axis=1), columns=['overnight model', 'trading day model']))
    print(f'overnight model total mean loss: {np.mean(loss_overnight)}')
    print(f'trading day model total mean loss: {np.mean(loss_now)}')
    print(f'forward->backward->optimize time cost: {t2-t1}')
    print()

overnight_map_in.clear_optimizer_memory()
overnight_ode.clear_optimizer_memory()
overnight_map_out.clear_optimizer_memory()
now_map_in.clear_optimizer_memory()
now_ode.clear_optimizer_memory()
now_map_out.clear_optimizer_memory()


# Prediction
over_maped_1=overnight_map_in.forward(np.tile(np.concatenate((new_qqqm[-1], qqqm_fft[-1], ndx_fft[-1])), (sim_num, 1)))
over_int=overnight_ode.forward(np.concatenate((over_maped_1[-1][-1], np.tile(vix[-1, 0:4], (sim_num, 1))), axis=1), np.tile(np.atleast_2d(time_interval[-1]/10), (sim_num, 1)), steps)
over_maped_2=overnight_map_out.forward(over_int+over_maped_1[-1][-1])

now_maped_1=now_map_in.forward(np.concatenate((over_maped_2[-2][-1], np.tile(qqqm_fft[-1], (sim_num, 1)), np.tile(ndx_fft[-1], (sim_num, 1))), axis=1))
now_int=now_ode.forward(np.concatenate((now_maped_1[-1][-1], np.tile(vix[-1, 0:4], (sim_num, 1))), axis=1), np.tile(np.atleast_2d(time_interval[-1]/10), (sim_num, 1)), steps)
now_maped_2=candle_layer(now_map_out.forward(now_int+now_maped_1[-1][-1])[-1][-1], k_scale_factor)

result=np.concatenate((now_maped_2, over_maped_2[-1][-1]), axis=1)
result[:, 3]=np.atleast_2d(np.exp(result[:, 3])*qqqm.to_numpy()[-2,0])
result[:, 0:3]=np.exp(result[:, 0:3])*result[:, 3].reshape(sim_num, 1)

print(f'{sim_num} times Monte Carlo Prediction...')
print(pd.DataFrame(result, columns=['Close', 'High', 'Low', 'Open']))
stactistic_data=np.array([[np.min(result[:, 0]), np.min(result[:, 1]), np.min(result[:, 2]), np.min(result[:, 3])],
                         [np.max(result[:, 0]), np.max(result[:, 1]), np.max(result[:, 2]), np.max(result[:, 3])],
                         [np.mean(result[:, 0]), np.mean(result[:, 1]), np.mean(result[:, 2]), np.mean(result[:, 3])],
                         [np.median(result[:, 0]), np.median(result[:, 1]), np.median(result[:, 2]), np.median(result[:, 3])]])
print(f'Stactistic data from {sim_num} times Monte Carlo Prediction')
print(pd.DataFrame(stactistic_data, columns=['Close', 'High', 'Low', 'Open'], index=['Min', 'Max', 'Mean', 'Median']))

overnight_ode.clean_memory()
now_ode.clean_memory()

# Prediction visualization
fig, axs=plt.subplots(2,3,figsize=(16,9))

axs[0,0].set_title('Loss Curve')
axs[0,0].set_xlabel('Epochs')
axs[0,0].set_ylabel('Total Mean Loss')
axs[0,0].plot(np.array(loss_list).T[0])
axs[0,0].plot(np.array(loss_list).T[1])
axs[0,0].legend(['overnight model', 'trading day model'], loc='best')

axs[0,1].set_title('QQQM Close Price')
axs[0,1].set_xlabel(f'mean: {stactistic_data[2, 0]}\nmedian: {stactistic_data[3, 0]}')
axs[0,1].set_ylabel('Close Price (USD)')
axs[0,1].scatter(np.arange(0,sim_num), result[:, 0])
axs[0,1].plot([0,sim_num], [stactistic_data[2, 0], stactistic_data[2, 0]], color='red', label='mean')
axs[0,1].plot([0,sim_num], [stactistic_data[3, 0], stactistic_data[3, 0]], color='yellow', label='median')
axs[0,1].legend(loc='best')

axs[0,2].set_title('QQQM High Price')
axs[0,2].set_xlabel(f'mean: {stactistic_data[2, 1]}\nmedian: {stactistic_data[3, 1]}')
axs[0,2].set_ylabel('Close Price (USD)')
axs[0,2].scatter(np.arange(0,sim_num), result[:, 1])
axs[0,2].plot([0,sim_num], [stactistic_data[2, 1], stactistic_data[2, 1]], color='red', label='mean')
axs[0,2].plot([0,sim_num], [stactistic_data[3, 1], stactistic_data[3, 1]], color='yellow', label='median')
axs[0,2].legend(loc='best')

axs[1,0].set_title('QQQM Low Price')
axs[1,0].set_xlabel(f'mean: {stactistic_data[2, 2]}\nmedian: {stactistic_data[3, 2]}')
axs[1,0].set_ylabel('Close Price (USD)')
axs[1,0].scatter(np.arange(0,sim_num), result[:, 2])
axs[1,0].plot([0,sim_num], [stactistic_data[2, 2], stactistic_data[2, 2]], color='red', label='mean')
axs[1,0].plot([0,sim_num], [stactistic_data[3, 2], stactistic_data[3, 2]], color='yellow', label='median')
axs[1,0].legend(loc='best')

axs[1,1].set_title('QQQM Open Price')
axs[1,1].set_xlabel(f'mean: {stactistic_data[2, 3]}\nmedian: {stactistic_data[3, 3]}')
axs[1,1].set_ylabel('Close Price (USD)')
axs[1,1].scatter(np.arange(0,sim_num), result[:, 3])
axs[1,1].plot([0,sim_num], [stactistic_data[2, 3], stactistic_data[2, 3]], color='red', label='mean')
axs[1,1].plot([0,sim_num], [stactistic_data[3, 3], stactistic_data[3, 3]], color='yellow', label='median')
axs[1,1].legend(loc='best')

fig.tight_layout(pad=2.5)
plt.show()