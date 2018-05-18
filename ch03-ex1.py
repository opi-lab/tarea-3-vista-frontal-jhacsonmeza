from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os


def make_homog(points):
      """ Convert a set of points (dim*n array) to
      homogeneous coordinates. """
      
      return np.vstack((points,np.ones((1,points.shape[1]))))


def Htransform(im, H, out_size):
  # Applies a homography transform to im
  pil_im = Image.fromarray(im)
  pil_size = out_size[1], out_size[0]
  return np.array(pil_im.transform(
    pil_size, Image.PERSPECTIVE, H.reshape(9)[0:8] / H[2,2], Image.LINEAR))


def H_from_points(x1, x2):
      n = len(x1.T)
      
      # condition points (important for numerical reasons)
      # --from points--
      m = np.mean(x1[:2], axis=1)
      maxstd = max(np.std(x1[:2], axis=1)) + 1e-9
      C1 = np.diag([1/maxstd, 1/maxstd, 1])
      C1[0][2] = -m[0]/maxstd
      C1[1][2] = -m[1]/maxstd
      x1 = np.dot(C1,x1)
      # --to points--
      m = np.mean(x2[:2], axis=1)
      maxstd = max(np.std(x2[:2], axis=1)) + 1e-9
      C2 = np.diag([1/maxstd, 1/maxstd, 1])
      C2[0][2] = -m[0]/maxstd
      C2[1][2] = -m[1]/maxstd
      x2 = np.dot(C2,x2)
      
      A = np.ones([3*n,9])
      zero = np.array([0,0,0])
      for i in range(n):            
            x = x1[:,i]
            A[3*i] = np.hstack([zero, -x, x2[1,i]*x])
            A[3*i+1] = np.hstack([x, zero, -x2[0,i]*x])
            A[3*i+2] = np.hstack([-x2[1,i]*x, x2[0,i]*x, zero])
            
      *_,Vh = np.linalg.svd(A)
      x = Vh.T[:,-1]
      H = x.reshape(3,3)
      
      # decondition
      H = np.dot(np.linalg.inv(C2),np.dot(H,C1))
      
      # normalize and return
      return H / H[2,2]

      
#im1 = np.array(Image.open(os.path.abspath('data/book_frontal.jpg')).\
#               convert('L'))
im1 = np.array(Image.open(os.path.abspath('data/book_perspective.jpg')).\
               convert('L'))

plt.figure('Imagen 1')
plt.axis('off'), plt.imshow(im1,cmap='gray')
x1 = np.array(plt.ginput(4,mouse_add=3,mouse_pop=1,mouse_stop=2)).T
plt.close()

h, w = 300, 400
x2 = np.array([[0,h,0,h],[0,0,w,w]])


xh1 = make_homog(x1)
xh2 = make_homog(x2)
H = H_from_points(xh2, xh1)
im_out = Htransform(im1, H, (w,h))

plt.figure()
plt.subplot(121), plt.imshow(im1,cmap='gray'), plt.axis('off')
plt.plot(x1[0],x1[1],'r.',markersize=6)
plt.subplot(122), plt.imshow(im_out,cmap='gray'), plt.axis('off')
plt.show()