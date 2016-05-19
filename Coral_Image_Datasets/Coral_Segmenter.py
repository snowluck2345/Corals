
# coding: utf-8

# In[1]:

import scipy.io as sio
import numpy
import os
import cv2
import sys
import re
import PIL
import Image
import glob
import math

import matplotlib
matplotlib.use('Agg')


from pylab import *

start_dir = os.getcwd()


# In[2]:

dir_contents = os.listdir(start_dir)

for index in range(len(dir_contents)):
    print dir_contents[index]

matching = [s for s in dir_contents if ".png" in s]


# In[3]:

size_image = []
for s in range(len(matching)):
    image = Image.open(matching[s])
    size_image.append(image.size)
    
print size_image


# In[4]:

same_size = []
print len(size_image)
print len(range(len(size_image)))
for i in range(len(size_image)):
    for j in range(len(size_image)):
        if i != j:
            temp = size_image[i] == size_image[j]
            if temp == True:
                same_size.append([i,j])


# In[5]:

def merge(lsts):
  sets = [set(lst) for lst in lsts if lst]
  merged = 1
  while merged:
    merged = 0
    results = []
    while sets:
      common, rest = sets[0], sets[1:]
      sets = []
      for x in rest:
        if x.isdisjoint(common):
          sets.append(x)
        else:
          merged = 1
          common |= x
      results.append(common)
    sets = results
  return sets

nice_image_sets = list(merge(same_size))
print nice_image_sets


# In[6]:

for j in range(len(nice_image_sets)):
    for i in range(len(list(nice_image_sets[j]))):
        print size_image[list(nice_image_sets[j])[i]]


# In[7]:

def image_segmenter_index_finder(base_image_size, min_overlap_pixels, sub_image_size):
    z = int(math.ceil(base_image_size/float(sub_image_size-min_overlap_pixels)))
    actual_overlap = (sub_image_size * z - base_image_size) / z
    crop_spot = [[None]*2 for _ in range(z)] 
    crop_spot[0][0] = 0
    crop_spot[0][1] = sub_image_size
    for i in range(1,z):
        crop_spot[i][0] = crop_spot[i - 1][1]  - actual_overlap
        crop_spot[i][1] = crop_spot[i][0] + sub_image_size
    crop_spot[z-1][0] = base_image_size - sub_image_size
    crop_spot[z-1][1] = base_image_size
    return crop_spot


# In[12]:

#current important vars:
#nice_image_sets: contains index of sets of images with same sizes
#size_image: contains sizes of images, order from directory, same order as matching 
#matching: list of all images in directory with .png suffix
os.chdir(start_dir)
#all: label/annotation, based on species, MANY MORE LABELS
#morphs = label/annotation, based on morphology
#raw = x
test_ratio = 0.20
train_ratio = 0.70
val_ratio = 0.10



#for j in range(len(nice_image_sets)):
#    for i in range(len(list(nice_image_sets[j]))):
i = 0
j = 0
num_images = 0

for j in range(len(nice_image_sets)):
    current_image = Image.open(matching[list(nice_image_sets[j])[i]])
    #imshow(current_image)

    current_dims = size_image[list(nice_image_sets[j])[i]]
    base_image_size = current_dims[0]
    min_overlap_pixels = 100 #total overlap in an image, both sides
    sub_image_size = 640 #square image
    segment_indexes_width = image_segmenter_index_finder(base_image_size, min_overlap_pixels, sub_image_size)

    current_dims = size_image[list(nice_image_sets[j])[i]]
    base_image_size = current_dims[1]
    min_overlap_pixels = 100 #total overlap in an image, both sides
    sub_image_size = 480 #square image
    segment_indexes_height = image_segmenter_index_finder(base_image_size, min_overlap_pixels, sub_image_size)

    #print segment_indexes_width
    #print segment_indexes_height

    #print len(segment_indexes_width)
    #print len(segment_indexes_height)

    num_images = len(segment_indexes_width) * len(segment_indexes_height) + num_images

num_train = int(num_images * train_ratio)
num_val = int(num_images * val_ratio)
num_test = int(num_images - num_val - num_train)

print num_train
print num_val
print num_test

#coord = [0,0]
pixel_points =[[0 for x in range(2)] for y in range(2)] 

counter_train = 0
counter_trainannot = 0
counter_val = 0
counter_valannot = 0
counter_test = 0
counter_testannot = 0

file_name_trainannot = []
file_name_valannot = []
file_name_testannot = []
file_name_train = []
file_name_val = []
file_name_test = []

val_crop_points = []
test_crop_points = []
train_crop_points = []



# In[13]:

for k in range(len(segment_indexes_width)):
        for m in range(len(segment_indexes_height)):
            pixel_points[0][0] = segment_indexes_width[k][0]
            pixel_points[0][1] = segment_indexes_height[m][0]
            pixel_points[1][0] = segment_indexes_width[k][1]
            pixel_points[1][1] = segment_indexes_height[m][1]


# In[14]:

print len(segment_indexes_width)
print len(segment_indexes_height)
print j


# In[15]:

#print (list(nice_image_sets[0]))
#print (list(nice_image_sets[1]))
#print (list(nice_image_sets[2]))
#print (list(nice_image_sets[3]))
#print (list(nice_image_sets[4]))


# In[ ]:

counter_train = 0
counter_trainannot = 0
counter_val = 0
counter_valannot = 0
counter_test = 0
counter_testannot = 0


for j in range(0,1): #in range(len(list(nice_image_sets))):
    for i in range(len(list(nice_image_sets[j]))):
        current_image = Image.open(matching[list(nice_image_sets[j])[i]])

        if 'MORPHS' in matching[list(nice_image_sets[j])[i]]:
            data = np.array(current_image)
            red, green, blue, alpha = data.T
            white_areas0 = (red == 82) & (blue == 78) & (green == 241)
            white_areas1 = (red == 255) & (blue == 254) & (green == 88)
            white_areas2 = (red == 255) & (blue == 0) & (green == 0)
            white_areas3 = (red == 10) & (blue == 243) & (green == 0)
            data[..., :-1][white_areas0.T] = (1, 1, 1) # Transpose back needed
            data[..., :-1][white_areas1.T] = (2, 2, 2) # Transpose back needed
            data[..., :-1][white_areas2.T] = (3, 3, 3) # Transpose back needed
            data[..., :-1][white_areas3.T] = (0, 0, 0) # Transpose back needed

            red, green, blue, alpha = data.T
            cv2.imwrite('temp.png', red.T)

            current_image = Image.open('temp.png')

        for k in range(len(segment_indexes_width)):
            for m in range(len(segment_indexes_height)):
                #print 'hi'
                pixel_points[0][0] = segment_indexes_width[k][0]
                pixel_points[0][1] = segment_indexes_height[m][0]
                pixel_points[1][0] = segment_indexes_width[k][1]
                pixel_points[1][1] = segment_indexes_height[m][1]

                #temp_points = copy.deepcopy(pixel_points)

                #for n in range(2): #width loop
                #    for o in range(2): #height loop
                #        coord[0] =  segment_indexes_width[k][n]
                #        coord[1] =  segment_indexes_height[m][o]
                #        if n== 0 and o == 0:
                #            print n
                #            print o
                #            print coord
                #            pixel_points[0] = coord
                #        if n== 1 and o == 1:
                #            print n
                #            print o
                #            print coord
                #            pixel_points[1] = coord
                        #coord[0] = width pixel
                        #coord[1] = heigh pixel
                        #loops top left pair, bottom left pair, top right pair, bottom right pair
                        #goes through columns of sub images
                #        print coord
                #print pixel_points
                image_cropped = current_image.crop((pixel_points[0][0], pixel_points[0][1]
                                                    , pixel_points[1][0], pixel_points[1][1])) #(x_top,y_top,x_bottom_y_bottom)
                file_name = matching[list(nice_image_sets[j])[i]] + '_small_' + str(k) + '_' + str(m) + '.png'


                if 'MORPHS' in file_name:
                    #print temp_points
                    if counter_trainannot < num_train:
                        #print pixel_points

                        #print temp_points
                        #print pixel_points
                        train_crop_points.append(str(pixel_points[:]))
                        print(train_crop_points[0])

                        image_cropped.save('./chopped/trainannot/' + file_name, "PNG")
                        file_name_trainannot.append('/chopped/trainannot/' + file_name)
                        counter_trainannot = counter_trainannot + 1
                    elif counter_valannot < num_val:
                        val_crop_points.append(str(pixel_points[:]))
                        image_cropped.save('./chopped/valannot/' + file_name, "PNG")
                        file_name_valannot.append('/chopped/valannot/' + file_name)
                        counter_valannot = counter_valannot + 1
                    elif counter_testannot < num_test:
                        test_crop_points.append(str(pixel_points[:]))
                        image_cropped.save('./chopped/testannot/' + file_name, "PNG")
                        file_name_testannot.append('/chopped/testannot/' + file_name)
                        counter_testannot = counter_testannot + 1
                if 'RAW' in file_name:
                    if counter_train < num_train:
                        image_cropped.save('./chopped/train/' + file_name, "PNG")
                        file_name_train.append('/chopped/train/' + file_name)
                        counter_train = counter_train + 1
                    elif counter_val < num_val:
                        image_cropped.save('./chopped/val/' + file_name, "PNG")
                        file_name_val.append('/chopped/val/' + file_name)
                        counter_val = counter_val + 1
                    elif counter_test < num_test:
                        image_cropped.save('./chopped/test/' + file_name, "PNG")
                        file_name_test.append('/chopped/test/' + file_name)
                        counter_test = counter_test + 1

                #if 'MORPHS' in file_name:
                #    image_cropped.save('./chopped/valannot/' + file_name, "PNG")
                #if 'RAW' in file_name:
                #    image_cropped.save('./chopped/val/' + file_name, "PNG")

                #if 'MORPHS' in file_name:
                #    image_cropped.save('./chopped/testannot/' + file_name, "PNG")
                #if 'RAW' in file_name:
                #    image_cropped.save('./chopped/test/' + file_name, "PNG")


 #   figure()
 #   imshow(image_cropped)


# In[ ]:

print train_crop_points


# In[ ]:

print file_name_trainannot


# In[ ]:

start_dir = os.getcwd()
print start_dir


# In[ ]:

i = 1
print start_dir + file_name_train[i] + ' ' + start_dir + file_name_trainannot[i] + ' ' + str(train_crop_points[i]) + '\n'


# In[ ]:

#name_list = full_path + space_list.tolist() + labels.tolist()


train_file = open("train.txt","wb")
for i in range(0, len(file_name_train)):
    train_file.write(start_dir + file_name_train[i] + ' ' + start_dir + file_name_trainannot[i] + '\n')
train_file.close()

test_file = open("test.txt","wb")
for i in range(0, len(file_name_test)):
    test_file.write(start_dir + file_name_test[i] + ' ' + start_dir + file_name_testannot[i]+ '\n')
test_file.close()

val_file = open("val.txt", "wb")
for i in range(0, len(file_name_val)):
    val_file.write(start_dir + file_name_val[i] + ' ' + start_dir + file_name_valannot[i]+ '\n')
val_file.close()

#now saving cropped names for restitiching
train_file = open("train_crop.txt","wb")
for i in range(0, len(file_name_train)):
    train_file.write(start_dir + file_name_train[i] + ' ' + start_dir + file_name_trainannot[i] + ' ' + str(train_crop_points[i]) + '\n')
train_file.close()

test_file = open("test_crop.txt","wb")
for i in range(0, len(file_name_test)):
    test_file.write(start_dir + file_name_test[i] + ' ' + start_dir + file_name_testannot[i] + ' ' + str(test_crop_points[i]) + '\n')
test_file.close()

val_file = open("val_crop.txt", "wb")
for i in range(0, len(file_name_val)):
    val_file.write(start_dir + file_name_val[i] + ' ' + start_dir + file_name_valannot[i] + ' ' + str(val_crop_points[i]) + '\n')
val_file.close()



# In[ ]:

j = 0
i = 0

im = Image.open(matching[list(nice_image_sets[j])[i]])
print len(filter(None, im.histogram()))
print filter(None, im.histogram())

temp = im.convert('LA')

len(filter(None, temp.histogram()))


# In[ ]:

print im.getcolors()


# In[ ]:

print temp.getcolors()


# In[ ]:

data = np.array(im)
red, green, blue, alpha = data.T
white_areas = (red == 82) & (blue == 78) & (green == 241)
data[..., :-1][white_areas.T] = (255, 255, 255) # Transpose back needed

im2 = Image.fromarray(data)
#imshow(im2)


# In[ ]:

#imshow(im)


# In[ ]:

data = np.array(im)
red, green, blue, alpha = data.T
white_areas0 = (red == 82) & (blue == 78) & (green == 241)
white_areas1 = (red == 255) & (blue == 254) & (green == 88)
white_areas2 = (red == 255) & (blue == 0) & (green == 0)
white_areas3 = (red == 10) & (blue == 243) & (green == 0)

255, 88, 254
255, 0, 0
10, 0, 243

data[..., :-1][white_areas0.T] = (1, 1, 1) # Transpose back needed
data[..., :-1][white_areas1.T] = (2, 2, 2) # Transpose back needed
data[..., :-1][white_areas2.T] = (3, 3, 3) # Transpose back needed
data[..., :-1][white_areas3.T] = (0, 0, 0) # Transpose back needed




im2 = Image.fromarray(data)
im2 = im2.convert('LA')
#imshow(im2)
im2.save('/home/dschreib/' + 'hi' + '.png', "PNG")


# In[ ]:

im2 = Image.fromarray(data)
red, green, blue, alpha = data.T
cv2.imwrite('/home/dschreib/' + 'test' + '.png',red.T)
current_image = Image.open('/home/dschreib/' + 'test' + '.png')
pixels =  (filter(None, current_image.histogram()))
total_pixels = sum(pixels)

print 1/(float(pixels[0]) / float(total_pixels))/6
print 1/(float(pixels[1]) / float(total_pixels))/6
print 1/(float(pixels[2]) / float(total_pixels))/6
print 1/(float(pixels[3]) / float(total_pixels))/6


current_image.save('/home/dschreib/' + 'test' + 'bob',"PNG")

print 5555555
print pixels
print 2222222
print pixels[0]
print pixels[1]
print pixels[2]
print pixels[3]
print total_pixels

# In[ ]:




# In[ ]:



