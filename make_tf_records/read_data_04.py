import numpy as np
file = 'GMIv5_ml1_06.bin'
data = np.fromfile(file, dtype=np.float32, count=-1)

file_size =104092648              # file size in bites 
el_per_rec=3377                   # number of elements in one record
n_rec     =file_size/el_per_rec/4 # this number of records in the file.

n_rec      = int(n_rec)
el_per_rec = int(el_per_rec)

#reshape the data into [n_rec,el_per_rec]
data_in= np.reshape(data,(n_rec,el_per_rec))


#initiate arrayas to store data
stra=np.zeros(n_rec)
conv=np.zeros(n_rec)
tbs =np.zeros((n_rec,15,25,9)) 

#populate each array
for rec in range(n_rec): # loop over records
    conv[rec]=data_in[rec,0] # convective fraction
    stra[rec]=data_in[rec,1] # stratiform fraction
    counter=2                # initiate counter to skip the first two elements
    for scan in range(9): # loop over scans
        for pix in range(25): # loop over pixels
            for ch in range(15): # loop over channals
                tbs[rec,ch,pix,scan]=data_in[rec,counter] # join the vale
                counter=counter+1

# get rid of missing elements 
  # declare
tbs_new             =np.zeros((n_rec,13,25,9)) 
  # populate
tbs_new[:,0:5,:,:]  =tbs[:,0:5,:,:]            
tbs_new[:,5:11,:,:] =tbs[:,6:12,:,:]
tbs_new[:,11:14,:,:]=tbs[:,13:16,:,:]
  # rename
tbs=tbs_new

# normalize
  # declare
tbs_norm            =np.zeros((n_rec,13,25,9)) 
  # populate with normalized values
max_tbs=np.amax(tbs) # max value of the array
for rec in range(n_rec):
    for ch in range(13):
        tbs_norm[rec,ch,:,:]=tbs[rec,ch,:,:] / max_tbs

# print something to check if reading and manipulating is done correctly
print(np.amin(tbs_norm))
print(np.amax(tbs_norm))

print(conv[9])
print(stra[9])
print(tbs[9,0,:,:])
# add for saving in the right format for matching cifar 10
t_conv = np.reshape(conv,(1,conv.size))
t_tbs_norm = np.reshape(tbs,(7706,13*25*9))
LL = np.concatenate((t_conv.T, t_tbs_norm), axis=1)  # this will make a list of arrays

ss = np.reshape(LL,-1)
ss1 = ss.astype(np.float32)
ss1.tofile('test_my_data_non_norm.bin')

# now lets read it back and print and make sure there is a match
test2 = np.fromfile('test_my_data_non_norm.bin', dtype='float32,(13,25,9)float32', count=-1)
x = test2[9][1]
print('-------------------------------')
print(x[0,0,:])
print('----------compare arrays-------')
print(tbs[9,0,0,:])
print('-------------------------------')
print('read label:',test2[9][0])
print('-------------------------------')
print('print label:',conv[9])
print('-------------------------------')

