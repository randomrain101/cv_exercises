#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
#
# --- aufgabenblatt 1 ---
#
def tiling(arr, x, y):
    temp = np.concatenate(tuple([arr])*x, axis=0)
    return np.concatenate(tuple([temp])*y, axis=1)

#%%
arr = plt.imread("test.png")

#%%
plt.imshow(arr)

# %%
arr2 = tiling(arr, 3, 4)
plt.imshow(arr2)

# %%
def crop(arr, x1, x2, y1, y2):
    return arr[x1:x2, y1:y2, :]

arr3 = crop(arr2, 50, -200, 100, -400)
plt.imshow(arr3)

#%%
#
# --- blatt 1 ---
#
pic = plt.imread("test.png")

#%%
plt.imshow(pic)
plt.show()
weights=[0.2989,0.5870,0.1140]
#weights=[1, 1, 0]
#weights=[1, 0, 0]
#weights=[0, 1, 0]
grayscaleimage=np.dot(pic[:, :, :3],weights)
plt.imshow(grayscaleimage,cmap=plt.get_cmap("gray"))

# %%
