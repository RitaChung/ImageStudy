import os
import pandas as pd
from dev_script import *

label_list = pd.read_csv('label_list.csv')

img_set = os.listdir()
img_collect = []
for file in img_set:
  if file.endswith('jpeg'):
    path = os.path.join(os.getcwd(),file)
    extractor = local_features_extractor(path)
    #print(file.replace('.jpeg',''))
    #print('image size = {0}'.format(extractor.toImage().shape))
    features = extractor.feature()
    #print('num of keypoints = ',features['num_of_keypoints'])
    img_collect.append(file)

result = pd.DataFrame({'queryImage':[],'databaseImage':[],'similarity':[]})
item = 0
for run in range(len(img_collect)):
  pic1 = img_collect[run]
  for run2 in range(run+1,len(img_collect)):
    pic2 = img_collect[run2]
    if(run2 <= len(img_collect)):
      extractor1 = local_features_extractor(pic1).feature()
      extractor2 = local_features_extractor(pic2).feature()
      #matchPoint = imageMatcher(extractor1,extractor2).matcher()
      similarity = imageMatcher(extractor1,extractor2).similarity()
      #print(str(pic1)+" with "+str(pic2)+" have "+str(similarity['num_of_valid_matches'])+" points matched and similarity is "+str(similarity['percent_of_matches']))
      item += 2
      result.loc[item-1]  = [extractor1['name'].replace('.jpeg',''), extractor2['name'].replace('.jpeg',''), similarity['percent_of_matches']]
      result.loc[item] = [extractor2['name'].replace('.jpeg',''), extractor1['name'].replace('.jpeg',''), similarity['percent_of_matches']]

temp = result.merge(label_list,left_on='databaseImage', right_on='id', how='left')
temp = temp.groupby(['queryImage','label']).mean('similarity')
print(temp)
