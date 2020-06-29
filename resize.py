import cv2
import matplotlib.pyplot as plt
img = 'img/4.jpg'
img = cv2.imread(img,0)
img = cv2.resize(img,(img.shape[1]//2,img.shape[0]//2))[400:650,:]
cv2.imwrite('debug/4.jpg',img)
print(img.shape)
plt.imshow(img,cmap = 'gray')
# cv2.waitKey()
plt.show()