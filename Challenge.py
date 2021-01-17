import numpy as np
import cv2 #only for loading images
import math

# Resource: http://www.graficaobscura.com/interp/index.html 
#https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#nearest
# https://en.wikipedia.org/wiki/Gaussian_blur
#https://courses.cs.vt.edu/~masc1044/L17-Rotation/ScalingNN.html

def brighten(image_path, alpha, save_loc, save_name):
    '''Brighten the image
    image_path: Path of the image
    alpha: brightness constant
    save_loc: Save location
    save_name: Save Name
    out = (1 - alpha)*in0 + alpha*in1
    in0 is black image or 0
    '''
    image = cv2.imread(image_path)
    image = np.uint16(image)
    image = (1 - alpha)*np.zeros_like(image) + alpha*image
    image = np.where(image<255,image,255)
    cv2.imwrite(save_loc+save_name+str(alpha)+".jpg", image)
    print("Done Brighten!")


image_path = "./input/princeton_small.jpg"
brighten(image_path, 0.0, "./output/", "princeton_small_brightness_")
brighten(image_path, 0.5, "./output/", "princeton_small_brightness_")
brighten(image_path, 2.0, "./output/", "princeton_small_brightness_")


def Luminace(image):
    image = np.uint16(image)
    return 0.30 * image[:,:,2] + 0.59 * image[:,:,1] + 0.11 * image[:,:,0]



def contrast(image_path, alpha, save_loc, save_name):
    '''
    Reference: 
    Average Luminance: https://in.mathworks.com/matlabcentral/answers/109662-how-to-write-the-code-for-the-average-luminance-of-input-image
    '''
    image = cv2.imread(image_path)
    image = np.uint16(image)
    img = Luminace(image)
    img[:] = (np.sum(img))/(img.shape[0]*img.shape[1])
    img2 = np.zeros_like(image)
    img2[:,:,0] = img
    img2[:,:,1] = img
    img2[:,:,2] = img
    image = (1 - alpha)*img2 + alpha*image
    cv2.imwrite(save_loc+save_name+str(alpha)+".jpg", image)
    print("Done Contrast!")
    

image_path = "./input/c.jpg"

contrast(image_path, -0.5, "./output/", "c_contrast_")
contrast(image_path, 0.0, "./output/", "c_contrast_")
contrast(image_path, 0.5, "./output/", "c_contrast_")
contrast(image_path, 2.0, "./output/", "c_contrast_")


def gauss_kernel(shape,sigma):
    size = shape[0]
    center= size//2
    kernel=np.zeros((size,size))
    for i in range(size):
        for j in range(size):            
            diff=((i-center)**2+(j-center)**2)
            kernel[i,j]=np.exp(-(diff)/(2*(sigma**2)))
    return kernel/np.sum(kernel)      
#     return kernel/np.sqrt(2*np.pi*(sigma**2))


def convolve2d(image, kernel):
    output = np.zeros_like(image)

    # Channel 1 
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    
    image_padded[1:-1, 1:-1] = image[:,:,0]
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x,0]=(kernel * image_padded[y: y+3, x: x+3]).sum()
            
    #Channel 2    
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image[:,:,1]
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x,1]=(kernel * image_padded[y: y+3, x: x+3]).sum()
    
    # Channel 3
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image[:,:,2]
    for x in range(image.shape[1]):
        for y in range(image.shape[0]):
            output[y, x,2]=(kernel * image_padded[y: y+3, x: x+3]).sum()  
                     

    return output


def blur(image_path, kernel_shape, sigma, save_loc, save_name):
    image = cv2.imread(image_path)
    image = np.uint16(image)
    cv2.imwrite(save_loc+save_name+str(sigma)+".jpg",convolve2d(image,gauss_kernel(kernel_shape,sigma)))
    print("Done Blur!")
    
    

blur("./input/princeton_small.jpg", (3,3), 0.125, "./output/", "blur_")
blur("./input/princeton_small.jpg", (3,3), 2, "./output/", "blur_")
blur("./input/princeton_small.jpg", (3,3), 8, "./output/", "blur_")


def sharpen(image_path, kernel_shape, alpha, save_loc, save_name):
    image = cv2.imread(image_path)
    image = np.uint16(image)
    blur = convolve2d(image,gauss_kernel(kernel_shape,2))
    image = (1 - alpha)*blur + alpha*image
    image = np.where(image<255,image,255)
    cv2.imwrite(save_loc+save_name+".jpg", image)
    print("Done sharpen!")

sharpen("./input/princeton_small.jpg", (3,3), 2.5, "./output/", "sharpen")


def detect_edge(image_path, kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]), save_loc=None, save_name=None):
    image = cv2.imread(image_path)
    image = np.uint16(image)
    image = convolve2d(image, kernel)
    cv2.imwrite(save_loc+save_name+".jpg", image)
    print("Done Edge_detection!")


detect_edge("./input/princeton_small.jpg", save_loc="./output/", save_name="edgedetect")


def Scaling(image_path, Sx, Sy, scaling_method, save_loc):
    image = cv2.imread(image_path)
    image = np.uint16(image)
    w, h = image.shape[:2]
    new_width = int(Sx*image.shape[0])
    new_height = int(Sy*image.shape[1])
    if scaling_method == 'point':
        scaled_image = np.zeros([new_width, new_height, 3])
        xScale = w/new_width; 
        yScale = h/new_height;
        for i in range(0,new_width):
            for j in range(0,new_height):
                scaled_image[i , j ]= image[1 + int(i * xScale),1 + int(j * yScale)]
                
        cv2.imwrite(save_loc+"scale_point.jpg", scaled_image)
        print("Done Point Scaling")
        
    if scaling_method == 'gaussian':
        scaled_image = np.zeros([new_width, new_height, 3])
        scaled_image = np.zeros([new_width, new_height, 3])
        xScale = w/new_width; 
        yScale = h/new_height;
        for i in range(0,new_width):
            for j in range(0,new_height):
                scaled_image[i , j ]= image[1 + int(i * xScale),1 + int(j * yScale)]
        scaled_image = convolve2d(scaled_image,gauss_kernel((3,3),1))
        cv2.imwrite(save_loc+"scale_gaussian.jpg", scaled_image)
        print("Done Gaussian Scaling")
                
    if scaling_method == "bilinear": 
        pass # Understood the theory but could not implement it. 
    
    
Scaling("./input/scaleinput.jpg", 0.3,0.3,"point","./output/")
Scaling("./input/scaleinput.jpg", 0.3,0.3,"gaussian","./output/")


def Composite(base_image_path, top_image_path, Alpha_channel_path):
    base = cv2.imread(base_image_path)
    top  = cv2.imread(top_image_path)
    mask = cv2.imread(Alpha_channel_path)
    image = ((mask/255)*top)+(base*(1-(mask/255)))
    cv2.imwrite("./output/composite.jpg",image)
    print("Done Composite")


Composite("./input/comp_background.jpg", "./input/comp_foreground.jpg", "./input/comp_mask.jpg")