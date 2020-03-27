from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='./data/motor_data'
dataType='val2017'
annFile=dataDir+'/via_export_coco.json'.format(dataDir,dataType)
coco=COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['motor']);
imgIds = coco.getImgIds(catIds=catIds );

imgIds = coco.getImgIds(imgIds = [20])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

I = io.imread(img['file_name'])
plt.axis('off')
plt.imshow(I)
plt.show() 

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
