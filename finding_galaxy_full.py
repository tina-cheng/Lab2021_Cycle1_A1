from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from scipy.optimize import curve_fit
from scipy.stats import norm
import copy
from tqdm import tqdm
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from scipy import ndimage
from collections import Counter
from itertools import groupby
import pandas as pd

hdulist=fits.open("A1_mosaic.fits")
hdulist.info()
#obtain headers
headers=hdulist[0].header
#obtain data
odata=hdulist[0].data
hdulist.close()    
#%%
with fits.open('masked.fits', mode='update') as file:
    data=copy.deepcopy(odata)
    file[0].data=data
    file.flush()
file.close()

#%%
h, w=data.shape[:2]
Y, X = np.ogrid[:h, :w]
center=[1428,3209]
radius=300
small_r=200
dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
mask = (dist_from_center >= small_r) & (dist_from_center <= radius)
#dist_from_center <= radius

with fits.open('masked.fits', mode='update') as file:
    data=copy.deepcopy(odata)
    data[mask]=0
    file[0].data=data
    file.flush()
file.close()

#%%   Masking
def circular_mask(h, w, center=[1000,1000], radius=40):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center >= radius
    return mask

with fits.open('masked.fits', mode='update') as file:
    data=copy.deepcopy(odata)
    #edges
    data[:,:120]=0
    data[:100,:]=0
    data[:,2430:]=0
    data[4500:,:]=0
    #large vertical bleeding in the middle
    data[:,1426:1448]=0
    #horizontal bleeding at the bottom
    data[426:428,1103:1652]=0
    data[428:430,1200:1595]=0
    data[430:435,1250:1595]=0
    data[432:457,1338:1510]=0
    data[457:472,1400:1470]=0
    data[457:472,1400:1470]=0
    data[457:472,1400:1470]=0
   
    data[314:318,1018:1703]=0
    data[318:322,1260:1570]=0
    data[322:332,1340:1516]=0
    data[332:352,1390:1478]=0
    data[352:372,1420:1458]=0
    data[372:382,1440:1458]=0
   
    data[214:250,1388:1476]=0
    data[250:265,1416:1456]=0
   
    data[124:127,1290:1524]=0
    data[116:124,1392:1464]=0
    data[127:130,1340:1500]=0
    data[130:140,1370:1480]=0
    data[140:150,1405:1456]=0
    data[150:163,1410:1428]=0
   
    data[0:10,1020:1700]=0
    data[1:3,970:1020]=0
    data[1:3,1700:1720]=0
    data[10:20,1330:1500]=0
    data[20:25,1390:1460]=0
    data[30:52,1414:1450]=0
    #bright stars
    data[3180:3415,770:779]=0
   
    data[2700:2835,970:976]=0
   
    data[2222:2360,902:908]=0
   
    data[3700:3805,2132:2136]=0
   
    center=[(1428,3209),(775,3318),(976,2774),(906,2284),(2133,2308),
            (2090,1426),(560,4096),(2134,3758),(1456,4032)]
    radius=[340,45,42,38,28,27,25,30,25]
#    masked=data.copy()
    for i in range(len(center)):
        h, w=data.shape[:2]
        mask=circular_mask(h, w, center=center[i], radius=radius[i])
        data[~mask]=0

    file[0].data=data
    file.flush()
file.close()
#%%
def find_galaxy(h, w, pos,radius=8):
    """
    given position of a maximum pixel, find all contributing pixels to that galaxy
    """
    Y, X = np.ogrid[:h, :w]
    center=pos
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
   
    mask = dist_from_center >= radius
    return mask

def find_background(h, w, pos,radius=12):
    """
    given position of a maximum pixel, find background to that galaxy
    """
    Y, X = np.ogrid[:h, :w]
    center=pos
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
   
    mask = dist_from_center >= radius
    return mask

def ring_background(h, w, center=[1000,1000], radius=12, small_r=8):
    """
    given position of a maximum pixel, find background values to that galaxy by using ring around maximum pixel
    """
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = (dist_from_center >= small_r) & (dist_from_center <= radius)
#    mask = dist_from_center >= radius
#    mask = ring >= dist_from_center
    return mask

def find_local_max(data_in):
    im = img_as_float(data_in)
    image_max = ndimage.maximum_filter(im, size=150, mode='nearest')   #am not sure what size does
    pos = peak_local_max(im, min_distance=10, threshold_rel=0.0862) #min dist between galaxies, make threshold smaller to get more galaxies
#    pos = peak_local_max(im, min_distance=10, threshold_rel=0.092)   
    #0.087 for 2765, 0.09 for 800
    return pos

def find_mode(data):
    #group most common output by frequency
    groups=groupby(Counter(data).most_common(), lambda x:x[1])
    # pick off the first group (highest frequency)
    modes=[val for val, count in next(groups)[1]]
    return modes

def magnitude_count(magnitude_data, magnitude):
    N=0
    for mag in magnitude_data:
        if mag < magnitude:
            N+=1
    return N

def check_mode(mod):
    if len(mod) == 1:
        mode = mod
    if mod == [0]:
#        new_arr=[v for v in sample[background] if v != 0]   #for sample
        new_arr=[v for v in data[background] if v != 0]     #for whole imgage
        new_mod=find_mode(new_arr)
        mode = new_mod
        if len(new_mod) == 1:
            mode = new_mod
        else:
            mode = [int(np.mean(new_mod))]
    else:
        mode = [int(np.mean(mod))]
    return mode

#%% small sample for testing
sample=copy.deepcopy(data[410:1120,650:1950])
with fits.open('samplearea.fits', mode='update') as file:
    file[0].data=sample
    file.flush()
file.close()
galaxies=[]     #all the found galaxies

#%% see how many galaxies/ their coords
pos = find_local_max(data)
#print(pos) #coords [y x]
print(len(pos)) #how many galaxies found

#%%  run to get more galaxies
h, w=sample.shape[:2]  
pos=find_local_max(sample)
galaxies=[]
bg=[]
print(len(pos))

#%%
#for i in range(len(pos)):  
#    r=8
#    bg_r=12
#    with fits.open('samplearea.fits', mode='update') as file:
#        file[0].data=sample
#        mask=find_galaxy(h, w, [pos[i][1],pos[i][0]], radius=r)
#        background=ring_background(h, w, [pos[i][1],pos[i][0]], radius=bg_r, small_r=r)
##        background=find_background(h, w, [pos[i][1],pos[i][0]])
#        mod=find_mode(sample[background])
#        mode=check_mode(mod)
#        
#        new_r=r      #only used if mode is too large
#        new_bg_r=bg_r
#        while mode > [3450]:    #if mode is too large, keep increasing ring
#            print(i)
##            sample[background]=0
#            print(mode)
#            print(sum(sample[~mask]))
#            new_r+=1
#            new_bg_r+=1
#            mask=find_galaxy(h, w, [pos[i][1],pos[i][0]], radius=new_r)
#            background=ring_background(h, w, [pos[i][1],pos[i][0]], radius=new_bg_r, small_r=new_r)
#            mod=find_mode(sample[background])
#            mode=check_mode(mod)
#            print('new',mode)
#            print('new', sum(sample[~mask])) 
#            
#        galaxies.append(sum(sample[~mask]))
#        bg.append(mode)
#        sample[background]=0
##        sample[~mask]=0
#        file.flush()
#    file.close()

#%% Sample
#h, w=sample.shape[:2]  
#pos=find_local_max(sample)
#galaxies=[]
#local_background=[]
#
#for i in range(len(pos)):  
##for i in range(20):
#    with fits.open('samplearea.fits', mode='update') as file:
#        file[0].data=sample
#        mask=find_galaxy(h, w, [pos[i][1],pos[i][0]])
#        background=find_background(h, w, [pos[i][1],pos[i][0]])
#        mod=find_mode(sample[~background])
##        print(len(mod))
#        if len(mod) == 1:
#            mode = mod
#        else:
#            mode = [int(np.mean(mod))]
##        print('mode', mode)
#        if mode == [0]:
#            new_arr=[v for v in sample[~background] if v != 0]
#            new_mod=find_mode(new_arr)
#            mode = new_mod
#        local_background.append(mode)
##        print(mode)
##        print(sum(mode*len(sample[~mask])))  
#        tot_val= sum(sample[~mask])-(sum(mode*len(sample[~mask])))
##        print('final count', tot_val)
#        if tot_val < 0:
#            print(i)       #prints the position index of the nagtive count galaxies, can use cell below to see it
#        galaxies.append(tot_val)
#        sample[~mask]=0
#        file.flush()
#    file.close()

#%%Whole image

h, w=data.shape[:2]  
pos=find_local_max(data)
galaxies=[]
local_background=[]
coord=[]
#neg=[]

for i in tqdm(range(len(pos))):  
    r=6
    bg_r=12
    with fits.open('masked.fits', mode='update') as file:
        file[0].data=data
        mask=find_galaxy(h, w, [pos[i][1],pos[i][0]], radius=r)
        background=ring_background(h, w, [pos[i][1],pos[i][0]], radius=bg_r, small_r=r)
        mod=find_mode(data[background])
        mode=check_mode(mod)
        
        #check if mode is less than threshold, if not aperture is too small for galaxy
        new_r=r      #only used if mode is too large
        new_bg_r=bg_r
        while mode > [3450]:    #if mode is too large, keep increasing ring
#            print(i)
            new_r+=1
            new_bg_r+=1
            mask=find_galaxy(h, w, [pos[i][1],pos[i][0]], radius=new_r)
            background=ring_background(h, w, [pos[i][1],pos[i][0]], radius=new_bg_r, small_r=new_r)
            mod=find_mode(data[background])
            mode=check_mode(mod)
#            print('new', sum(sample[~mask])) 
        
#        data[background]=0
        
        tot_val= sum(data[~mask])-(sum(mode*len(data[~mask])))
        if tot_val > 0:
            local_background.append(mode)
            galaxies.append(tot_val)
            coord.append(pos[i])
#        local_background.append(mode)
#        tot_val= sum(data[~mask])-(sum(mode*len(data[~mask])))
#        if tot_val < 0:
#            neg.append(i)       #prints the position index of the nagtive count galaxies, can use cell below to see it
#        galaxies.append(tot_val)
#        data[~mask]=0
        file.flush()
    file.close()

#print(len(neg))

#%%  calibrating fluxes
background=np.concatenate(local_background)
x_pos=[]
y_pos=[]
for i in range(len(coord)):
    x_pos.append(coord[i][1])
    y_pos.append(coord[i][0])

ZP=hdulist[0].header['MAGZPT']
error=hdulist[0].header['MAGZRR']
mag=ZP-2.5*np.log10(galaxies)

#%%
plt.hist(background, bins=100)
#%%   change pos index in this cell to see position of negative count galaxies in ds9
#with fits.open('samplearea.fits', mode='update') as file:
#    file[0].data=sample
#    mask=find_galaxy(h, w, [pos[15][1],pos[15][0]])
#    background=find_background(h, w, [pos[15][1],pos[15][0]])
#    mod=find_mode(sample[~background])
#    if len(mod) == 1:
#        mode = mod
#    else:
#        mode = [int(np.mean(mod))]
#
#    if mode == [0]:
#        new_arr=[v for v in sample[~background] if v != 0]
#        new_mod=find_mode(new_arr)
#        mode = new_mod
#    local_background.append(mode)
#
#    tot_val= sum(sample[~mask])-(sum(mode*len(sample[~mask])))
#    galaxies.append(tot_val)
#    sample[~mask]=0
#    file.flush()
#file.close()

#%%
#np.savetxt(fname='catalogue_sample', X=(x_pos, y_pos, galaxies,local_background),
#           fmt='%.4e', delimiter=',')

df = pd.DataFrame(x_pos, columns=['x'])
df['y']= y_pos
df['mag']= mag
df['bck']= background
#%%
#df.to_csv('im_800.csv', index=False)
df.to_csv('ad_r6_b12_9130.csv')

#%%
index, xcoord, ycoord, cal_mag, background_count = np.loadtxt('ad_r6_b12_9130.csv',
                                                              skiprows=1, unpack=True, delimiter=',')

#index2, xcoord2, ycoord2, cal_mag2, background_count2 = np.loadtxt('im_6_12_2765.csv',
#                                                              skiprows=1, unpack=True, delimiter=',')

#%%   Plot
x = np.arange(10, 22, 0.1)
y = []
y_err = []


for magnitude in x:
    count = magnitude_count(cal_mag, magnitude)
    y.append(np.log10(count))
    count_err= ((2.5/(np.log(10)*count))**2) * count
    z_err = (0.02 ** 2)
    y_err.append(np.sqrt(count_err+ z_err))

plt.figure(1)
plt.errorbar(x, y, yerr=y_err, color='blue', ecolor='lightgray', elinewidth=1, 
             capsize=1, label='9130 galaxies')

#x = np.arange(10, 22, 0.1)
#y = []
#y_err = []
#
#for magnitude in x:
#    count = magnitude_count(cal_mag2, magnitude)
#    y.append(np.log10(count))
#    count_err= ((2.5/(np.log(10)*count))**2) * count
#    z_err = (0.02 ** 2)
#    y_err.append(np.sqrt(count_err+ z_err))
#
#plt.figure(1)
#plt.errorbar(x, y, yerr=y_err, color='green', ecolor='lightgray', elinewidth=1, 
#             capsize=1, label='2765 galaxies')
plt.legend()
plt.show()
#%%

plt.figure(2)
x_fit= np.arange(10, 18, 0.1)
y_fit=[]
for magnitude in x_fit:
    count = magnitude_count(cal_mag, magnitude)
    y_fit.append(np.log10(count))
m, c =np.polyfit(x_fit,y_fit,1)
print(m,c)
fit, cov=np.polyfit(x_fit, y_fit, 1, cov=True)
error=np.sqrt(np.diag(cov))
print(error)

x_f= np.arange(16, 18, 0.1)
y_f=[]
for magnitude in x_f:
    count = magnitude_count(cal_mag, magnitude)
    y_f.append(np.log10(count))
m, c =np.polyfit(x_f,y_f,1)
print(m,c)
fit_1, cov_1=np.polyfit(x_f, y_f, 1, cov=True)
error=np.sqrt(np.diag(cov_1))
print(error)

plt.xlabel('Apparent magnitude, m (a.u)')
plt.errorbar(x, y, yerr=y_err, color='grey', ecolor='lightgray', elinewidth=1, 
             capsize=1, label='data and error bar')
plt.ylabel('Log(N(m)) (a.u)')
#plt.plot(x, y,"x")
long=np.arange(15,18,0.1)
pY_1=np.poly1d(fit_1)
plt.plot(long, pY_1(long), color='red',label='linear regression for above 16')

pY=np.poly1d(fit)
plt.plot(x_fit, pY(x_fit), color='green',label='linear regression for all data')

plt.legend()
plt.show()