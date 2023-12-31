from dataset import Human36M
import matplotlib.pyplot as plt
import numpy as np
# import utils.aug_utils as aug_util


dataset = Human36M(data_dir='./',mode='train')
joint_img = dataset.datalist[12]['joint_img']
joint_cam = dataset.datalist[12]['joint_cam']

# print(dataset)
joint_img = np.array([[ 473.6541  , 444.88696 ,5203.882  ],
 [ 500.99265  ,447.99222 ,5251.0825 ],
 [ 479.83392 , 530.7903  ,5448.807  ],
 [ 506.216   , 622.7914  ,5400.138  ],
 [ 445.81503 , 441.72485 ,5156.681  ],
 [ 456.16092 , 537.1746  ,5244.681  ],
 [ 467.204   , 634.1008 , 5290.7637 ],
 [ 488.14777 , 397.20282 ,5123.8906 ],
 [ 480.90833 , 339.61743 ,5074.6045 ],
 [ 478.3514  , 317.69827 ,5130.165  ],
 [ 485.61975 , 296.07672 ,5058.5986 ],
 [ 453.8017  , 359.16052 ,5050.592  ],
 [ 429.87268 , 415.51346 ,5134.1772 ],
 [ 412.80936 , 452.7784  ,5307.6274 ],
 [ 508.13968 , 355.92737 ,5140.2725 ],
 [ 520.3363  , 413.175   ,5257.7383 ],
 [ 515.4759  , 456.40582 ,5421.0684 ]])
joint_cam = np.array([[-176.73077 , -321.04865 , 5203.882   ],
 [ -52.961914 ,-309.70453 , 5251.0825  ],
 [-155.64156  ,  73.07175 , 5448.807   ],
 [ -29.831573 , 506.78442 , 5400.138   ],
 [-300.49985  ,-332.39282 , 5156.681   ],
 [-258.24048  ,  99.60901 , 5244.681   ],
 [-209.48436  , 548.8338  , 5290.7637  ],
 [-109.15762 , -529.7282  , 5123.8906  ],
 [-140.19118 , -780.1214  , 5074.6045  ],
 [-153.1819  , -886.97614 , 5130.165   ],
 [-118.93483 , -970.22833 , 5058.5986  ],
 [-259.08997 , -690.1336  , 5050.592   ],
 [-370.67087 , -448.59937 , 5134.1772  ],
 [-462.28662 , -290.82953 , 5307.6274  ],
 [ -19.760376, -716.91815 , 5140.2725  ],
 [  35.791595, -470.14502 , 5257.7383  ],
 [  13.892456 ,-279.85297 , 5421.0684  ]])

cam_x = joint_cam[:,0]
cam_y = joint_cam[:,1]
cam_z = joint_cam[:,2]

img_x = joint_img[:,0]
img_y = joint_img[:,1]
img_z = joint_img[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cam_x, cam_y, cam_z, c='b', marker='o')
ax.scatter(img_x, img_y, img_z, c='r', marker='o')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()