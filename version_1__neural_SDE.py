import numpy as np # hamiltonian neural network with diffusion and latent space

dt=np.float64
# You can modify the number of neurons but number of layers
M1=[6, 100, 10] # <- map-in layer structure
S=[6, 25, 5, 25, 5] # <- P and Q MLP layer structure
D=[S[-1]*2+1, 20, 5] # <- diffusion layer structure
M2=[10+2, 5] # <- map-out layer structure

#map-in layer weights and biases
map1_w1=np.random.normal(0,np.sqrt(2/M1[0]), (M1[1], M1[0])).astype(dt)
map1_b1=np.zeros((1, M1[1]), dtype=dt)

map1_w2=np.random.normal(0,np.sqrt(2/M1[1]), (M1[2], M1[1])).astype(dt)
map1_b2=np.zeros((1, M1[2]), dtype=dt)

#hamiltonian P term neural network weights and biases
p_w1=np.random.normal(0, np.sqrt(2/S[0]), (S[1], S[0])).astype(dt)
p_b1=np.zeros((1, S[1]), dtype=dt)

p_w2=np.random.normal(0, np.sqrt(2/S[1]), (S[2], S[1])).astype(dt)
p_b2=np.zeros((1, S[2]), dtype=dt)

p_w3=np.random.normal(0, np.sqrt(2/S[2]), (S[3], S[2])).astype(dt)
p_b3=np.zeros((1, S[3]), dtype=dt)

p_w4=np.random.normal(0, np.sqrt(2/S[3]), (S[4], S[3])).astype(dt)
p_b4=np.zeros((1, S[4]), dtype=dt)

#hamiltonian Q term neural network weights and biases
q_w1=np.random.normal(0, np.sqrt(2/S[0]), (S[1], S[0])).astype(dt)
q_b1=np.zeros((1, S[1]), dtype=dt)

q_w2=np.random.normal(0, np.sqrt(2/S[1]), (S[2], S[1])).astype(dt)
q_b2=np.zeros((1, S[2]), dtype=dt)

q_w3=np.random.normal(0, np.sqrt(2/S[2]), (S[3], S[2])).astype(dt)
q_b3=np.zeros((1, S[3]), dtype=dt)

q_w4=np.random.normal(0, np.sqrt(2/S[3]), (S[4], S[3])).astype(dt)
q_b4=np.zeros((1, S[4]), dtype=dt)

#diffusion neural network weights and biases
diff_w1=np.random.normal(0, np.sqrt(2/D[0]), (D[1], D[0])).astype(dt)
diff_b1=np.zeros((1, D[1]), dtype=dt)

diff_w2=np.random.normal(0, np.sqrt(2/D[1]), (D[2], D[1])).astype(dt)
diff_b2=np.zeros((1, D[2]), dtype=dt)

#map-out layer weights and biases
map2_w1=np.random.normal(0,np.sqrt(2/M2[0]), (M2[1], M2[0])).astype(dt)
map2_b1=np.zeros((1, M2[1]), dtype=dt)

# spectral normalization vector
p_u1=np.random.normal(0, np.sqrt(2/S[1]), (S[1], 1))
p_u2=np.random.normal(0, np.sqrt(2/S[3]), (S[3], 1))

q_u1=np.random.normal(0, np.sqrt(2/S[1]), (S[1], 1))
q_u2=np.random.normal(0, np.sqrt(2/S[3]), (S[3], 1))


def SN(w, u, gamma=1, initialize=False): # spectral normalization
  times=3 if initialize else 1
  for _ in range(times):
    v=np.dot(w.T,u)
    v/=np.linalg.norm(v)+1e-8

    u=np.dot(w, v)
    u/=np.linalg.norm(u)+1e-8

  sigma=np.dot(u.T, np.dot(w,v))
  return gamma*(w/sigma), u, sigma

def d_softplus(In): # derivative of softplus
  return (np.exp(In.astype(dt))/(1+np.exp(In.astype(dt))))

def map_1(In): # map-in layer
  z1=In@map1_w1.T+map1_b1
  a1=np.log(1+np.exp(z1.astype(dt)))

  z2=a1@map1_w2.T+map1_b2

  return z1, a1, z2

def map_2(In): # map-out layer
  return In@map2_w1.T+map2_b1

def mlp_p(In, time): # P MLP
  In=np.concatenate((In, np.ones((len(In), 1))*time), axis=1)

  z1=In@npm_1.T+p_b1
  a1=np.log(1+np.exp(z1.astype(dt)))

  z2=a1@p_w2.T+p_b2
  a2=np.log(1+np.exp(z2.astype(dt)))

  z3=a2@npm_2.T+p_b3
  a3=np.log(1+np.exp(z3.astype(dt)))

  z4=a3@p_w4.T+p_b4

  return [z1, a1, z2, a2, z3, a3, z4]

def grad_mlp_p(In, adjoint, time): # gradient of P MLP
  layers=mlp_p(In, time)

  g_b4=adjoint
  g_w4=np.einsum('ij,ik->ijk', g_b4, layers[5])

  g_b3=g_b4@p_w4*d_softplus(layers[4])
  g_w3=np.einsum('ij,ik->ijk', g_b3, layers[3])

  g_b2=g_b3@npm_2*d_softplus(layers[2])
  g_w2=np.einsum('ij,ik->ijk', g_b2, layers[1])

  g_b1=g_b2@p_w2*d_softplus(layers[0])
  g_w1=np.einsum('ij,ik->ijk', g_b1, np.concatenate((In, np.ones((len(In), 1))*time), axis=1))

  new_adjoint=g_b1@npm_1

  return [g_b4, g_w4, g_b3, g_w3, g_b2, g_w2, g_b1, g_w1, new_adjoint]

def mlp_q(In, time): # Q MLP
  In=np.concatenate((In, np.ones((len(In), 1))*time), axis=1)

  z1=In@nqm_1.T+q_b1
  a1=np.log(1+np.exp(z1.astype(dt)))

  z2=a1@q_w2.T+q_b2
  a2=np.log(1+np.exp(z2.astype(dt)))

  z3=a2@nqm_2.T+q_b3
  a3=np.log(1+np.exp(z3.astype(dt)))

  z4=a3@q_w4.T+q_b4

  return [z1, a1, z2, a2, z3, a3, z4]

def grad_mlp_q(In, adjoint, time): # gradient of Q MLP
  layers=mlp_q(In, time)

  g_b4=adjoint
  g_w4=np.einsum('ij,ik->ijk', g_b4, layers[5])

  g_b3=g_b4@p_w4*d_softplus(layers[4])
  g_w3=np.einsum('ij,ik->ijk', g_b3, layers[3])

  g_b2=g_b3@nqm_2*d_softplus(layers[2])
  g_w2=np.einsum('ij,ik->ijk', g_b2, layers[1])

  g_b1=g_b2@p_w2*d_softplus(layers[0])
  g_w1=np.einsum('ij,ik->ijk', g_b1, np.concatenate((In, np.ones((len(In), 1))*time), axis=1))

  new_adjoint=g_b1@nqm_1

  return [g_b4, g_w4, g_b3, g_w3, g_b2, g_w2, g_b1, g_w1, new_adjoint]

def diffusion(In, time): # Diffusion
  In=np.concatenate((In, np.ones((len(In), 1))*time), axis=1)

  z1=In@diff_w1.T+diff_b1
  a1=np.log(1+np.exp(z1.astype(dt)))

  z2=a1@diff_w2.T+diff_b2
  a2=np.log(1+np.exp(z2.astype(dt)))

  return [z1, a1, z2, a2]

def grad_diffusion(In, adjoint, time, noise, time_delta): # gradient of Diffusion
  layers=diffusion(In, time)

  g_diff_b2=(adjoint*noise*time_delta)*d_softplus(layers[2])
  g_diff_w2=np.einsum('ij,ik->ijk', g_diff_b2, layers[1])

  g_diff_b1=g_diff_b2@diff_w2*d_softplus(layers[0])
  g_diff_w1=np.einsum('ij,ik->ijk', g_diff_b1, np.concatenate((In, np.ones((len(In), 1))*time), axis=1))

  new_adjoint=g_diff_b1@diff_w1

  return [g_diff_b2, g_diff_w2, g_diff_b1, g_diff_w1, new_adjoint]

def hamiltonian(In, time, steps): # The ODE solver
  value=[In[:, 0:5], In[:, 5:]]
  delta_t=time/steps
  diffusion_result=[]
  noises=[]
  for i in range(0,steps):
    start_t=i*delta_t
    mid_t=start_t+delta_t/2
    end_t=start_t+delta_t

    p_half=value[0]-(delta_t/2)*mlp_q(value[1], start_t)[-1]
    q_full=value[1]+delta_t*mlp_p(p_half, mid_t)[-1]
    p_full=p_half-(delta_t/2)*mlp_q(q_full, end_t)[-1]

    noise=np.random.normal(0,1, (len(p_half), D[-1]))
    noises.append(noise)

    diffusion_term=diffusion(np.concatenate((p_half, q_full), axis=1), mid_t)[-1]
    diffusion_result.append(diffusion_term)

    p_full=p_full+diffusion_term*noise*np.sqrt(delta_t.astype(dt))

    value[0]=p_full
    value[1]=q_full

  value[0]=np.concatenate((value[0], np.ones((len(value[0]), 1))), axis=1)
  value[1]=np.concatenate((value[1], np.ones((len(value[1]), 1))), axis=1)

  return np.concatenate((value[0], value[1]), axis=1), noises, diffusion_result

def reverse_hamiltonian(In, adjoint, gradient_list, noise, diffusion_result, time, steps): # The reverse ODE solver
  p_full, q_full=In[:, 0:5], In[:, 6:-1]
  ap, aq=adjoint[:, :6], adjoint[:, 6:]
  gradients=gradient_list
  delta_t=time/steps

  for i in reversed(range(0,steps)):
    start_t=i*delta_t
    mid_t=start_t+(delta_t/2)
    end_t=start_t+delta_t

    p_full=p_full-diffusion_result[i]*noise[i]*np.sqrt(delta_t.astype(dt))

    p_half=p_full+(delta_t/2)*mlp_q(q_full, end_t)[-1]

    diff_gradient=grad_diffusion(np.concatenate((p_half, q_full), axis=1), ap[:, :5], mid_t, noise[i], np.sqrt(delta_t.astype(dt))) # i used p_half for calculating the diffusion term in forward pass
    ap[:, :5]+=diff_gradient[-1][:, 0:5] # adding diffusion gradients
    aq[:, :5]+=diff_gradient[-1][:, 5:-1] # adding diffusion gradients

    gradients[0]+=diff_gradient[0].astype(dt) # adding diffusion params gradients
    gradients[1]+=diff_gradient[1].astype(dt) # adding diffusion params gradients
    gradients[2]+=diff_gradient[2].astype(dt) # adding diffusion params gradients
    gradients[3]+=diff_gradient[3].astype(dt) # adding diffusion params gradients

    #calculating the Q MLP gradient
    q_gradient=grad_mlp_q(q_full, ap[:, 0:5], end_t)
    aq+=(delta_t/2).astype(dt)*q_gradient[-1].astype(dt)
    gradients[4]+=(delta_t/2).astype(dt)*q_gradient[0].astype(dt)
    gradients[5]+=np.atleast_3d(delta_t/2).astype(dt)*q_gradient[1].astype(dt)
    gradients[6]+=(delta_t/2).astype(dt)*q_gradient[2].astype(dt)
    gradients[7]+=np.atleast_3d(delta_t/2).astype(dt)*q_gradient[3].astype(dt)
    gradients[8]+=(delta_t/2).astype(dt)*q_gradient[4].astype(dt)
    gradients[9]+=np.atleast_3d(delta_t/2).astype(dt)*q_gradient[5].astype(dt)
    gradients[10]+=(delta_t/2).astype(dt)*q_gradient[6].astype(dt)
    gradients[11]+=np.atleast_3d(delta_t/2).astype(dt)*q_gradient[7].astype(dt)

    q_full=q_full-delta_t*mlp_p(p_half, mid_t)[-1]
    #calculating the P MLP gradient
    p_gradient=grad_mlp_p(p_half, aq[:, :5], mid_t)
    ap+=delta_t.astype(dt)*p_gradient[-1].astype(dt)
    gradients[12]+=delta_t.astype(dt)*p_gradient[0].astype(dt)
    gradients[13]+=np.atleast_3d(delta_t).astype(dt)*p_gradient[1].astype(dt)
    gradients[14]+=delta_t.astype(dt)*p_gradient[2].astype(dt)
    gradients[15]+=np.atleast_3d(delta_t).astype(dt)*p_gradient[3].astype(dt)
    gradients[16]+=delta_t.astype(dt)*p_gradient[4].astype(dt)
    gradients[17]+=np.atleast_3d(delta_t).astype(dt)*p_gradient[5].astype(dt)
    gradients[18]+=delta_t.astype(dt)*p_gradient[6].astype(dt)
    gradients[19]+=np.atleast_3d(delta_t).astype(dt)*p_gradient[7].astype(dt)

    p_full=p_half+(delta_t/2)*mlp_q(q_full, start_t)[-1]
    #calculating the Q MLP gradient
    q_gradient=grad_mlp_q(q_full, ap[:, 0:5], start_t)
    aq+=(delta_t/2).astype(dt)*q_gradient[-1].astype(dt)
    gradients[4]+=(delta_t/2).astype(dt)*q_gradient[0].astype(dt)
    gradients[5]+=np.atleast_3d(delta_t/2).astype(dt)*q_gradient[1].astype(dt)
    gradients[6]+=(delta_t/2).astype(dt)*q_gradient[2].astype(dt)
    gradients[7]+=np.atleast_3d(delta_t/2).astype(dt)*q_gradient[3].astype(dt)
    gradients[8]+=(delta_t/2).astype(dt)*q_gradient[4].astype(dt)
    gradients[9]+=np.atleast_3d(delta_t/2).astype(dt)*q_gradient[5].astype(dt)
    gradients[10]+=(delta_t/2).astype(dt)*q_gradient[6].astype(dt)
    gradients[11]+=np.atleast_3d(delta_t/2).astype(dt)*q_gradient[7].astype(dt)

  #gradient mean accumulation
  final_gradients=[]
  for j in range(0,len(gradients)):
    if gradients[j].ndim==2:
      gradients[j]=np.einsum('ij->j', gradients[j])/len(gradients[j])
    else:
      gradients[j]=np.einsum('ijk->jk', gradients[j])/len(gradients[j])

    # l2 gradient clipping
    norm=np.linalg.norm(gradients[j])
    if norm>threshold:
      gradients[j]=(threshold/norm)*gradients[j]

    final_gradients.append(np.ravel(gradients[j]))

  #print(f'Maximum reverse residuals: {np.max(np.array(np.concatenate((p_full,q_full), axis=1)-maped_1[-1]))}') # <- check for the biggest residual between maped input and reversed result

  #return a vectorized gradient, easy for adam optimizer calculation
  return np.concatenate(final_gradients), np.concatenate((ap[:, 0:5], aq[:, 0:5]), axis=1)

def huber(In, answer, h): # huber loss
  mse_signal=(np.absolute(answer-In)<=h)
  mse=(1/2)*((answer-In)**2)*mse_signal

  mae_signal=(np.absolute(answer-In)>h)
  mae=h*(np.absolute(answer-In)-(1/2)*h)*mae_signal
  return mse+mae

def d_huber(In, answer, h): # derivative of huber loss
  d_mse_signal=(np.absolute(answer-In)<=h)
  d_mse=-(answer-In)*d_mse_signal

  d_mae_signal=(np.absolute(answer-In)>h)
  d_mae=-h*np.sign(answer-In)*d_mae_signal

  return (d_mae+d_mse)/(5*len(In))


def adam(In, mean, variance, b1, b2, time): # adam optimizer
  mean=b1*mean+(1-b1)*In.astype(dt)
  variance=b2*variance+(1-b2)*(In.astype(dt)**2)
  new_m=mean/(1-b1**time)
  new_v=variance/(1-b2**time)

  return new_m/(np.sqrt(new_v)+1e-8), mean, variance


def data_solver(In):
  open=np.exp(In[:, 0].astype(dt)/10)*solve[0]
  high=np.exp(In[:, 1].astype(dt)/10)*solve[1]
  low=np.exp(In[:, 2].astype(dt)/10)*solve[2]
  close=np.exp(In[:, 3].astype(dt)/10)*solve[3]
  volumn=np.exp(In[:, 4].astype(dt)*15)-1

  return np.array([f'{np.min(open)} ~ {np.max(open)}', f'{np.min(high)} ~ {np.max(high)}', f'{np.min(low)} ~{np.max(low)}', f'{np.min(close)} ~ {np.max(close)}', f'{np.min(volumn)} ~ {np.max(volumn)}'])

#you can set the hyper paramaters
num=128 # <-batch size
data=train_data[0:-1]
time=time_interval[0:-1]
answer=train_answer[0:-1]
test=np.tile(test_data, (1000,1))
print(test)
steps=5            # steps for leapfrog integration
threshold=0.05     # gradient clipping threshold
lr=0.005           # learning rate
beta_1=0.95        # beta for adam mean
beta_2=0.99        # beta for adam variance
lam=0.005          # weight decay lambda
gamma=3            # spectral normalization gamma

# spilting whole data into multiple batch data
batch=[]
for i in range(0, int(len(data)/num)):
  batch.append(num)
if len(data)-len(batch)*num!=0:
  batch.append(len(data)-len(batch)*num)
print(batch)
print()

adam_mean=np.zeros(map2_b1.size+map2_w1.size+
                   diff_b2.size+diff_w2.size+diff_b1.size+diff_w1.size+
                   (p_b4.size+p_w4.size+p_b3.size+p_w3.size+p_b2.size+p_w2.size+p_b1.size+p_w1.size)*2+
                   map1_b2.size+map1_w2.size+map1_b1.size+map1_w1.size)

adam_variance=np.zeros(map2_b1.size+map2_w1.size+
                       diff_b2.size+diff_w2.size+diff_b1.size+diff_w1.size+
                       (p_b4.size+p_w4.size+p_b3.size+p_w3.size+p_b2.size+p_w2.size+p_b1.size+p_w1.size)*2+
                       map1_b2.size+map1_w2.size+map1_b1.size+map1_w1.size)

for t in range(0,20):
  lower=0
  upper=batch[0]
  print(f'the {t+1} th epochs...........................')
  for i in range(0,len(batch)):

    # forward pass
    p1=SN(p_w1, p_u1, gamma, True if t==0 else False)
    p2=SN(p_w3, p_u2, gamma, True if t==0 else False)


    q1=SN(q_w1, q_u1, gamma, True if t==0 else False)
    q2=SN(q_w3, q_u2, gamma, True if t==0 else False)

    if True:
      npm_1, p_u1=p1[0], p1[1]
      npm_2, p_u2=p2[0], p2[1]
      nqm_1, q_u1=q1[0], q1[1]
      nqm_2, q_u2=q2[0], q2[1]
    else:
      npm_1, p_u1=p_w1, p1[1]
      npm_2, p_u2=p_w3, p2[1]
      nqm_1, q_u1=q_w1, q1[1]
      nqm_2, q_u2=q_w3, q2[1]

    maped_1=map_1(data[lower:upper])
    ode=hamiltonian(maped_1[-1], time[lower:upper], steps)
    maped_2=map_2(ode[0])

    #print(maped_2) # output of forward pass
    #print()

    # loss
    loss=huber(maped_2, answer[lower:upper], 0.1)
    print(f'{batch[i]} data: {np.mean(loss)}')

    # backward pass
    g_map2_b1=d_huber(maped_2, answer[lower:upper], 0.1)
    g_map2_w1=np.einsum('ij,ik->ijk', g_map2_b1, ode[0])
    adjoint=g_map2_b1@map2_w1

    gradient_list=[np.zeros((batch[i], D[2])), np.zeros((batch[i], D[2], D[1])), np.zeros((batch[i], D[1])), np.zeros((batch[i], D[1], D[0])),
                   np.zeros((batch[i], S[4])), np.zeros((batch[i], S[4], S[3])), np.zeros((batch[i], S[3])), np.zeros((batch[i], S[3], S[2])),np.zeros((batch[i], S[2])), np.zeros((batch[i], S[2], S[1])),np.zeros((batch[i], S[1])), np.zeros((batch[i], S[1], S[0])),
                  np.zeros((batch[i], S[4])), np.zeros((batch[i], S[4], S[3])), np.zeros((batch[i], S[3])), np.zeros((batch[i], S[3], S[2])),np.zeros((batch[i], S[2])), np.zeros((batch[i], S[2], S[1])),np.zeros((batch[i], S[1])), np.zeros((batch[i], S[1], S[0]))]

    gradients=reverse_hamiltonian(ode[0], adjoint, gradient_list, ode[1], ode[2], time[lower:upper], steps)
    g_map1_b2=gradients[1]
    g_map1_w2=np.einsum('ij,ik->ijk', g_map1_b2, maped_1[1])
    g_map1_b1=g_map1_b2@map1_w2*d_softplus(maped_1[0])
    g_map1_w1=np.einsum('ij,ik->ijk', g_map1_b1, data[lower:upper])


    gradient_list=np.concatenate((np.ravel(np.einsum('ij->j', g_map2_b1)/batch[i]), np.ravel(np.einsum('ijk->jk', g_map2_w1)/batch[i]),
                                 gradients[0],
                                 np.ravel(np.einsum('ij->j', g_map1_b2)/batch[i]), np.ravel(np.einsum('ijk->jk', g_map1_w2)/batch[i]),np.ravel(np.einsum('ij->j', g_map1_b1)/batch[i]), np.ravel(np.einsum('ijk->jk', g_map1_w1)/batch[i])))

    #print(f'Max gradient: {np.max(gradient_list)}')

    new_gradients=adam(gradient_list, adam_mean, adam_variance, beta_1, beta_2, t+1)
    gradient_list=list(new_gradients[0])
    adam_mean=new_gradients[1]
    adam_variance=new_gradients[2]

    # weight update with weight decay
    map2_b1-=lr*np.array([gradient_list.pop(0) for _ in range(0,map2_b1.size)], dtype=dt)
    map2_w1-=lr*(np.array([gradient_list.pop(0) for _ in range(0,map2_w1.size)], dtype=dt).reshape(M2[1], M2[0])+lam*map2_w1)

    diff_b2-=lr*np.array([gradient_list.pop(0) for _ in range(0,diff_b2.size)], dtype=dt)
    diff_w2-=lr*(np.array([gradient_list.pop(0) for _ in range(0,diff_w2.size)], dtype=dt).reshape(D[2], D[1])+lam*diff_w2)
    diff_b1-=lr*np.array([gradient_list.pop(0) for _ in range(0,diff_b1.size)], dtype=dt)
    diff_w1-=lr*(np.array([gradient_list.pop(0) for _ in range(0,diff_w1.size)], dtype=dt).reshape(D[1], D[0])+lam*diff_w1)

    p_b4-=lr*np.array([gradient_list.pop(0) for _ in range(0,p_b4.size)], dtype=dt)
    p_w4-=lr*(np.array([gradient_list.pop(0) for _ in range(0,p_w4.size)], dtype=dt).reshape(S[4], S[3])+lam*p_w4)
    p_b3-=lr*np.array([gradient_list.pop(0) for _ in range(0,p_b3.size)], dtype=dt)
    p_w3-=lr*(np.array([gradient_list.pop(0) for _ in range(0,p_w3.size)], dtype=dt).reshape(S[3], S[2])+lam*p_w3)
    p_b2-=lr*np.array([gradient_list.pop(0) for _ in range(0,p_b2.size)], dtype=dt)
    p_w2-=lr*(np.array([gradient_list.pop(0) for _ in range(0,p_w2.size)], dtype=dt).reshape(S[2], S[1])+lam*p_w2)
    p_b1-=lr*np.array([gradient_list.pop(0) for _ in range(0,p_b1.size)], dtype=dt)
    p_w1-=lr*(np.array([gradient_list.pop(0) for _ in range(0,p_w1.size)], dtype=dt).reshape(S[1], S[0])+lam*p_w1)

    q_b4-=lr*np.array([gradient_list.pop(0) for _ in range(0,q_b4.size)], dtype=dt)
    q_w4-=lr*(np.array([gradient_list.pop(0) for _ in range(0,q_w4.size)], dtype=dt).reshape(S[4], S[3])+lam*q_w4)
    q_b3-=lr*np.array([gradient_list.pop(0) for _ in range(0,q_b3.size)], dtype=dt)
    q_w3-=lr*(np.array([gradient_list.pop(0) for _ in range(0,q_w3.size)], dtype=dt).reshape(S[3], S[2])+lam*q_w3)
    q_b2-=lr*np.array([gradient_list.pop(0) for _ in range(0,q_b2.size)], dtype=dt)
    q_w2-=lr*(np.array([gradient_list.pop(0) for _ in range(0,q_w2.size)], dtype=dt).reshape(S[2], S[1])+lam*q_w2)
    q_b1-=lr*np.array([gradient_list.pop(0) for _ in range(0,q_b1.size)], dtype=dt)
    q_w1-=lr*(np.array([gradient_list.pop(0) for _ in range(0,q_w1.size)], dtype=dt).reshape(S[1], S[0])+lam*q_w1)

    map1_b2-=lr*np.array([gradient_list.pop(0) for _ in range(0,map1_b2.size)], dtype=dt)
    map1_w2-=lr*(np.array([gradient_list.pop(0) for _ in range(0,map1_w2.size)], dtype=dt).reshape(M1[2], M1[1])+lam*map1_w2)
    map1_b1-=lr*np.array([gradient_list.pop(0) for _ in range(0,map1_b1.size)], dtype=dt)
    map1_w1-=lr*(np.array([gradient_list.pop(0) for _ in range(0,map1_w1.size)], dtype=dt).reshape(M1[1], M1[0])+lam*map1_w1)

    lower+=batch[i]
    upper+=batch[i]

    #print(f'Sigma term in spectral normalization: {p1[2]}, {p2[2]}, {q1[2]}, {q2[2]}') # check for sigma term used in spectral normalization
  print()

# testing
p1=SN(p_w1, p_u1, gamma, True if t==0 else False)
p2=SN(p_w3, p_u2, gamma, True if t==0 else False)


q1=SN(q_w1, q_u1, gamma, True if t==0 else False)
q2=SN(q_w3, q_u2, gamma, True if t==0 else False)

if True:
  npm_1, p_u1=p1[0], p1[1]
  npm_2, p_u2=p2[0], p2[1]
  nqm_1, q_u1=q1[0], q1[1]
  nqm_2, q_u2=q2[0], q2[1]
else:
  npm_1, p_u1=p_w1, p1[1]
  npm_2, p_u2=p_w3, p2[1]
  nqm_1, q_u1=q_w1, q1[1]
  nqm_2, q_u2=q_w3, q2[1]

maped_1=map_1(test)
ode=hamiltonian(maped_1[-1], np.array(test[:, -1]).reshape(1000,1), steps)
maped_2=map_2(ode[0])

solved_result=data_solver(maped_2)
print('Today QQQM stocks...')
print(f'Open: {solved_result[0]}')
print(f'high: {solved_result[1]}')
print(f'low: {solved_result[2]}')
print(f'close: {solved_result[3]}')
print(f'volumn: {solved_result[4]}')
