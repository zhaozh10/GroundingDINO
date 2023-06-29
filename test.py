from copy import deepcopy
import json
from tqdm import tqdm
from copy import deepcopy

# data=json.load(open('./train_cxr_object_coco.json'))
# # ann=info['annotations']

# # data=deepcopy(info)
# for i, elem in enumerate(tqdm(data['annotations'])):
#     elem['iscrowd']=0
#     elem['area']=elem['bbox'][2]*elem['bbox'][3]

# # info=deepcopy(data)
# with open('./train_cxr_object_coco.json','w')as f:
#     json.dump(data,f,indent=4)
# print("hold")

data=json.load(open('./train_cxr_object_coco.json'))
# ann=info['annotations']

# data=deepcopy(info)
for i, elem in enumerate(tqdm(data['annotations'])):
    elem['iscrowd']=0
    elem['area']=elem['bbox'][2]*elem['bbox'][3]

# info=deepcopy(data)
with open('./train_cxr_object_coco.json','w')as f:
    json.dump(data,f,indent=4)
print("hold")
