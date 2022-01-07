import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
env = 'pointmass_rooms'

'''''
#door
df_gcsl_nt = pd.read_csv('data/example/door/gcsl_mt1_sto/2022_01_01_13_34_26/progress.csv')
df_gcsl_n = pd.read_csv('data/example/door/gcsl_m1_sto/2022_01_02_00_11_54/progress.csv')
df_gcsl = pd.read_csv('data/example/door/gcsl_o/2021_12_24_01_24_32/progress.csv')
df_gcsl_s = pd.read_csv('data/example/door/gcsl_sm_sto/2021_12_25_23_39_19/progress.csv')
'''''
'''''
#pusher
df_gcsl_nt = pd.read_csv('data/example/pusher/gcsl_mt1_sto/2022_01_01_13_34_25/progress.csv')
df_gcsl_n = pd.read_csv('data/example/pusher/gcsl_m1_sto/2022_01_02_00_11_53/progress.csv')
df_gcsl = pd.read_csv('data/example/pusher/gcsl_o/2021_12_24_01_24_31/progress.csv')
df_gcsl_s = pd.read_csv('data/example/pusher/gcsl_sm_sto/2021_12_25_23_39_19/progress.csv')
'''''
'''''
#lunar
df_gcsl_nt = pd.read_csv('data/example/lunar/gcsl_mt1_sto/2022_01_01_13_34_25/progress.csv')
df_gcsl_n = pd.read_csv('data/example/lunar/gcsl_m1_sto/2022_01_02_00_11_54/progress.csv')
df_gcsl = pd.read_csv('data/example/lunar/gcsl_o/2021_12_24_01_24_30/progress.csv')
df_gcsl_s = pd.read_csv('data/example/lunar/gcsl_sm_sto/2021_12_25_23_39_19/progress.csv')
'''''
#'''''
#pointmass_rooms
df_gcsl_nt = pd.read_csv('data/example/pointmass_rooms/gcsl_mt1_sto/2022_01_01_13_34_25/progress.csv')
df_gcsl_n = pd.read_csv('data/example/pointmass_rooms/gcsl_m1_sto/2022_01_02_00_11_53/progress.csv')
df_gcsl = pd.read_csv('data/example/pointmass_rooms/gcsl_o/2021_12_24_01_24_31/progress.csv')
df_gcsl_s = pd.read_csv('data/example/pointmass_rooms/gcsl_sm_sto/2021_12_25_23_39_19/progress.csv')
#'''''
'''''
#pointmass_empty
df_gcsl_nt = pd.read_csv('data/example/pointmass_empty/gcsl_mt1_sto/2022_01_01_13_34_25/progress.csv')
df_gcsl_n = pd.read_csv('data/example/pointmass_empty/gcsl_m1_sto/2022_01_02_00_11_53/progress.csv')
df_gcsl = pd.read_csv('data/example/pointmass_empty/gcsl_o/2021_12_24_01_24_31/progress.csv')
df_gcsl_s = pd.read_csv('data/example/pointmass_empty/gcsl_sm_sto/2021_12_25_23_39_19/progress.csv')
'''''

nt_time = df_gcsl_nt['timesteps'].values
nt_success = df_gcsl_nt['Eval success ratio'].values
nt_avg_dist = df_gcsl_nt['Eval avg final dist'].values
nt_loss = df_gcsl_nt['policy loss'].values

n_time = df_gcsl_n['timesteps'].values
n_success = df_gcsl_n['Eval success ratio'].values
n_avg_dist = df_gcsl_n['Eval avg final dist'].values
n_loss = df_gcsl_n['policy loss'].values
#n_dist_mean = df_gcsl_n['Eval final puck distance Mean'].values
#n_dist_std = df_gcsl_n['Eval final puck distance Std'].values

time = df_gcsl['timesteps'].values
success = df_gcsl['Eval success ratio'].values
avg_dist = df_gcsl['Eval avg final dist'].values
loss = df_gcsl['policy loss'].values
#dist_mean = df_gcsl['Eval final puck distance Mean'].values
#dist_std = df_gcsl['Eval final puck distance Std'].values

s_time = df_gcsl_s['timesteps'].values
s_success = df_gcsl_s['Eval success ratio'].values
s_avg_dist = df_gcsl_s['Eval avg final dist'].values
s_loss = df_gcsl_s['policy loss'].values
########################################################################################################################
#Seed_1

'''''
#door
df_gcsl_nt_1 = pd.read_csv('data/example/door/gcsl_mt1_sto/2022_01_01_16_31_46/progress.csv')
df_gcsl_n_1 = pd.read_csv('data/example/door/gcsl_m1_sto/2022_01_02_03_01_45/progress.csv')
df_gcsl_1 = pd.read_csv('data/example/door/gcsl_o/2021_12_24_03_35_38/progress.csv')
df_gcsl_s_1 = pd.read_csv('data/example/door/gcsl_sm_sto/2021_12_26_03_12_47/progress.csv')
'''''
'''''
#pusher
df_gcsl_nt_1 = pd.read_csv('data/example/pusher/gcsl_mt1_sto/2022_01_01_16_31_46/progress.csv')
df_gcsl_n_1 = pd.read_csv('data/example/pusher/gcsl_m1_sto/2022_01_02_03_01_45/progress.csv')
df_gcsl_1 = pd.read_csv('data/example/pusher/gcsl_o/2021_12_24_03_35_37/progress.csv')
df_gcsl_s_1 = pd.read_csv('data/example/pusher/gcsl_sm_sto/2021_12_26_03_12_47/progress.csv')
'''''
'''''
#lunar
df_gcsl_nt_1 = pd.read_csv('data/example/lunar/gcsl_mt1_sto/2022_01_01_16_31_45/progress.csv')
df_gcsl_n_1 = pd.read_csv('data/example/lunar/gcsl_m1_sto/2022_01_02_03_01_45/progress.csv')
df_gcsl_1 = pd.read_csv('data/example/lunar/gcsl_o/2021_12_24_03_35_37/progress.csv')
df_gcsl_s_1 = pd.read_csv('data/example/lunar/gcsl_sm_sto/2021_12_26_03_12_46/progress.csv')
'''''
#'''''
#pointmass_rooms
df_gcsl_nt_1 = pd.read_csv('data/example/pointmass_rooms/gcsl_mt1_sto/2022_01_01_16_31_46/progress.csv')
df_gcsl_n_1 = pd.read_csv('data/example/pointmass_rooms/gcsl_m1_sto/2022_01_02_03_01_44/progress.csv')
df_gcsl_1 = pd.read_csv('data/example/pointmass_rooms/gcsl_o/2021_12_24_03_35_37/progress.csv')
df_gcsl_s_1 = pd.read_csv('data/example/pointmass_rooms/gcsl_sm_sto/2021_12_26_03_12_46/progress.csv')
#'''''
'''''
#pointmass_empty
df_gcsl_nt_1 = pd.read_csv('data/example/pointmass_empty/gcsl_mt1_sto/2022_01_01_16_31_45/progress.csv')
df_gcsl_n_1 = pd.read_csv('data/example/pointmass_empty/gcsl_m1_sto/2022_01_02_03_01_44/progress.csv')
df_gcsl_1 = pd.read_csv('data/example/pointmass_empty/gcsl_o/2021_12_24_03_35_37/progress.csv')
df_gcsl_s_1 = pd.read_csv('data/example/pointmass_empty/gcsl_sm_sto/2021_12_26_03_12_47/progress.csv')
'''''

nt_time_1 = df_gcsl_nt_1['timesteps'].values
nt_success_1 = df_gcsl_nt_1['Eval success ratio'].values
nt_avg_dist_1 = df_gcsl_nt_1['Eval avg final dist'].values
nt_loss_1 = df_gcsl_nt_1['policy loss'].values

n_time_1 = df_gcsl_n_1['timesteps'].values
n_success_1 = df_gcsl_n_1['Eval success ratio'].values
n_avg_dist_1 = df_gcsl_n_1['Eval avg final dist'].values
n_loss_1 = df_gcsl_n_1['policy loss'].values
#n_dist_mean = df_gcsl_n['Eval final puck distance Mean'].values
#n_dist_std = df_gcsl_n['Eval final puck distance Std'].values

time_1 = df_gcsl_1['timesteps'].values
success_1 = df_gcsl_1['Eval success ratio'].values
avg_dist_1 = df_gcsl_1['Eval avg final dist'].values
loss_1 = df_gcsl_1['policy loss'].values
#dist_mean = df_gcsl['Eval final puck distance Mean'].values
#dist_std = df_gcsl['Eval final puck distance Std'].values

s_time_1 = df_gcsl_s_1['timesteps'].values
s_success_1 = df_gcsl_s_1['Eval success ratio'].values
s_avg_dist_1 = df_gcsl_s_1['Eval avg final dist'].values
s_loss_1 = df_gcsl_s_1['policy loss'].values
#######################################################################################################################
#Seed 2
'''''
#door
df_gcsl_nt_2 = pd.read_csv('data/example/door/gcsl_mt1_sto/2022_01_01_19_29_01/progress.csv')
df_gcsl_n_2 = pd.read_csv('data/example/door/gcsl_m1_sto/2022_01_02_05_50_55/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/door/gcsl_o/2021_12_24_05_47_13/progress.csv')
df_gcsl_s_2 = pd.read_csv('data/example/door/gcsl_sm_sto/2021_12_26_06_45_17/progress.csv')
'''''
'''''
#pusher
df_gcsl_nt_2 = pd.read_csv('data/example/pusher/gcsl_mt1_sto/2022_01_01_19_29_00/progress.csv')
df_gcsl_n_2 = pd.read_csv('data/example/pusher/gcsl_m1_sto/2022_01_02_05_50_54/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/pusher/gcsl_o/2021_12_24_05_47_12/progress.csv')
df_gcsl_s_2 = pd.read_csv('data/example/pusher/gcsl_sm_sto/2021_12_26_06_45_16/progress.csv')
'''''
'''''
#lunar
df_gcsl_nt_2 = pd.read_csv('data/example/lunar/gcsl_mt1_sto/2022_01_01_19_29_00/progress.csv')
df_gcsl_n_2 = pd.read_csv('data/example/lunar/gcsl_m1_sto/2022_01_02_05_50_54/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/lunar/gcsl_o/2021_12_24_05_47_12/progress.csv')
df_gcsl_s_2 = pd.read_csv('data/example/lunar/gcsl_sm_sto/2021_12_26_06_45_16/progress.csv')
'''''
#'''''
#pointmass_rooms
df_gcsl_nt_2 = pd.read_csv('data/example/pointmass_rooms/gcsl_mt1_sto/2022_01_01_19_29_00/progress.csv')
df_gcsl_n_2 = pd.read_csv('data/example/pointmass_rooms/gcsl_m1_sto/2022_01_02_05_50_54/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/pointmass_rooms/gcsl_o/2021_12_24_05_47_12/progress.csv')
df_gcsl_s_2 = pd.read_csv('data/example/pointmass_rooms/gcsl_sm_sto/2021_12_26_06_45_16/progress.csv')
#'''''
'''''
#pointmass_empty
df_gcsl_nt_2 = pd.read_csv('data/example/pointmass_empty/gcsl_mt1_sto/2022_01_01_19_29_00/progress.csv')
df_gcsl_n_2 = pd.read_csv('data/example/pointmass_empty/gcsl_m1_sto/2022_01_02_05_50_54/progress.csv')
df_gcsl_2 = pd.read_csv('data/example/pointmass_empty/gcsl_o/2021_12_24_05_47_11/progress.csv')
df_gcsl_s_2 = pd.read_csv('data/example/pointmass_empty/gcsl_sm_sto/2021_12_26_06_45_16/progress.csv')
'''''

nt_time_2 = df_gcsl_nt_2['timesteps'].values
nt_success_2 = df_gcsl_nt_2['Eval success ratio'].values
nt_avg_dist_2 = df_gcsl_nt_2['Eval avg final dist'].values
nt_loss_2 = df_gcsl_nt_2['policy loss'].values

n_time_2 = df_gcsl_n_2['timesteps'].values
n_success_2 = df_gcsl_n_2['Eval success ratio'].values
n_avg_dist_2 = df_gcsl_n_2['Eval avg final dist'].values
n_loss_2 = df_gcsl_n_2['policy loss'].values
#n_dist_mean = df_gcsl_n['Eval final puck distance Mean'].values
#n_dist_std = df_gcsl_n['Eval final puck distance Std'].values

time_2 = df_gcsl_2['timesteps'].values
success_2 = df_gcsl_2['Eval success ratio'].values
avg_dist_2 = df_gcsl_2['Eval avg final dist'].values
loss_2 = df_gcsl_2['policy loss'].values
#dist_mean = df_gcsl['Eval final puck distance Mean'].values
#dist_std = df_gcsl['Eval final puck distance Std'].values

s_time_2 = df_gcsl_s_2['timesteps'].values
s_success_2 = df_gcsl_s_2['Eval success ratio'].values
s_avg_dist_2 = df_gcsl_s_2['Eval avg final dist'].values
s_loss_2 = df_gcsl_s_2['policy loss'].values
#######################################################################################################################
#Average and Std deviations by seeds
stack_success = np.vstack((success,success_1,success_2))
stack_s_success = np.vstack((s_success,s_success_1,s_success_2))
stack_n_success = np.vstack((n_success,n_success_1,n_success_2))
stack_nt_success = np.vstack((nt_success,nt_success_1,nt_success_2))

a_success = np.mean(stack_success,axis = 0)
a_s_success = np.mean(stack_s_success,axis = 0)
a_n_success = np.mean(stack_n_success,axis = 0)
a_nt_success = np.mean(stack_nt_success,axis = 0)

std_success = np.std(stack_success,axis = 0)
std_s_success = np.std(stack_s_success,axis = 0)
std_n_success = np.std(stack_n_success,axis = 0)
std_nt_success = np.std(stack_nt_success,axis = 0)

stack_avg_dist = np.vstack((avg_dist,avg_dist_1,avg_dist_2))
stack_s_avg_dist = np.vstack((s_avg_dist,s_avg_dist_1,s_avg_dist_2))
stack_n_avg_dist = np.vstack((n_avg_dist,n_avg_dist_1,n_avg_dist_2))
stack_nt_avg_dist = np.vstack((nt_avg_dist,nt_avg_dist_1,nt_avg_dist_2))

a_avg_dist = np.mean(stack_avg_dist,axis = 0)
a_s_avg_dist = np.mean(stack_s_avg_dist,axis = 0)
a_n_avg_dist = np.mean(stack_n_avg_dist,axis = 0)
a_nt_avg_dist = np.mean(stack_nt_avg_dist,axis = 0)

std_avg_dist = np.std(stack_avg_dist,axis = 0)
std_s_avg_dist = np.std(stack_s_avg_dist,axis = 0)
std_n_avg_dist = np.std(stack_n_avg_dist,axis = 0)
std_nt_avg_dist = np.std(stack_nt_avg_dist,axis = 0)

stack_loss = np.vstack((loss,loss_1,loss_2))
stack_s_loss = np.vstack((s_loss,s_loss_1,s_loss_2))
stack_n_loss = np.vstack((n_loss,n_loss_1,n_loss_2))
stack_nt_loss = np.vstack((nt_loss,nt_loss_1,nt_loss_2))

a_loss = np.mean(stack_loss,axis = 0)
a_s_loss = np.mean(stack_s_loss,axis = 0)
a_n_loss = np.mean(stack_n_loss,axis = 0)
a_nt_loss = np.mean(stack_nt_loss,axis = 0)

std_loss = np.std(stack_loss,axis = 0)
std_s_loss = np.std(stack_s_loss,axis = 0)
std_n_loss = np.std(stack_n_loss,axis = 0)
std_nt_loss = np.std(stack_nt_loss,axis = 0)


## Plot 1

plt.plot(time,a_success,'m', label = 'GCSL')
plt.fill_between(time,a_success - std_success , a_success + std_success,color = 'm' , alpha = 0.2)
plt.plot(time,a_n_success,'r',label = 'GCSL_Norm')
plt.fill_between(time,a_n_success - std_n_success , a_n_success + std_n_success,color = 'r' , alpha = 0.2)
plt.plot(time,a_nt_success,'g',label = 'GCSL_Norm_T')
plt.fill_between(time,a_nt_success - std_nt_success , a_nt_success + std_nt_success,color = 'g' , alpha = 0.2)
plt.plot(time,a_s_success,'b',label = 'GCSL_Shared')
plt.fill_between(time,a_s_success - std_s_success , a_s_success + std_s_success,color = 'b' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Success Ratio')
plt.legend()
plt.title('Success Ratio -' + env )
plt.grid()
#plt.show()
plt.savefig('plots_new/'+env+'_sto/success.jpg')
plt.close()

## Plot 2
plt.plot(time,a_avg_dist,'m', label = 'GCSL')
plt.fill_between(time,a_avg_dist - std_avg_dist , a_avg_dist + std_avg_dist,color = 'm' , alpha = 0.2)
plt.plot(time,a_n_avg_dist,'r',label = 'GCSL_Norm')
plt.fill_between(time,a_n_avg_dist - std_n_avg_dist , a_n_avg_dist + std_n_avg_dist,color = 'r' , alpha = 0.2)
plt.plot(time,a_nt_avg_dist,'g',label = 'GCSL_Norm_T')
plt.fill_between(time,a_nt_avg_dist - std_nt_avg_dist , a_nt_avg_dist + std_nt_avg_dist,color = 'g' , alpha = 0.2)
plt.plot(time,a_s_avg_dist,'b',label = 'GCSL_Shared')
plt.fill_between(time,a_s_avg_dist - std_s_avg_dist , a_s_avg_dist + std_s_avg_dist,color = 'b' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Average Distance')
plt.legend()
plt.title('Average Distance -' + env )
plt.grid()
#plt.show()
plt.savefig('plots_new/'+env+'_sto/avg_dist.jpg')
plt.close()

## Plot 3

plt.plot(time,a_loss,'m', label = 'GCSL')
plt.fill_between(time,a_loss - std_loss , a_loss + std_loss,color = 'm' , alpha = 0.2)
plt.plot(time,a_n_loss,'r',label = 'GCSL_Norm')
plt.fill_between(time,a_n_loss - std_n_loss , a_n_loss + std_n_loss,color = 'r' , alpha = 0.2)
plt.plot(time,a_nt_loss,'g',label = 'GCSL_Norm_T')
plt.fill_between(time,a_nt_loss - std_nt_loss , a_nt_loss + std_nt_loss,color = 'g' , alpha = 0.2)
plt.plot(time,a_s_loss,'b',label = 'GCSL_Shared')
plt.fill_between(time,a_s_loss - std_s_loss , a_s_loss + std_s_loss,color = 'b' , alpha = 0.2)
plt.xlabel('TimeSteps')
plt.ylabel('Policy Loss')
plt.legend()
plt.title('Policy Loss -' + env )
plt.grid()
#plt.show()
plt.savefig('plots_new/'+env+'_sto/loss.jpg')
plt.close()



