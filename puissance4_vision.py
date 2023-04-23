# Modules
import cv2
import numpy as np
import math
from collections import namedtuple, OrderedDict
import Puissance4_intel
import matplotlib.pyplot as plt
MIN_CONTOUR_WIDTH=10
MIN_CONTOUR_HEIGHT=10
MAX_CONTOUR_WIDTH=75
MAX_CONTOUR_HEIGHT=75
COLUMNS=7
ROWS=6
# Functions

class BBox(namedtuple('BBox', 'x y w h')):
    @property
    def area(self):
        return self.w * self.h

    @property
    def center(self):
        return self.x + self.w // 2, self.y + self.h // 2

    def __str__(self):
        return "(x=%d, y=%d, w=%d, h=%d)" % (self.x, self.y, self.w, self.h)


ShapeData = namedtuple('ShapeData', 'bbox contour')

def imgshow(name,img):
    cv2.imshow(name,img)
    cv2.moveWindow(name,200,200)
    cv2.waitKey(0)

# Read and process image
file_name = "./B3.jpg"
img = cv2.imread(file_name)
mini_width=100
min_height=100
max_width=150
max_height=150

def crop_shear(img):
  
    
    
    val_rec=[]
     # get the image shape
    rows, cols = img.shape[:2]
    taille=img.shape
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (36 , 0 , 0), (86, 255 ,255))
    imask = mask>0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]

    contours, hierarchy= cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        bound=cv2.boundingRect(contour)
        #print(bound)
        x,y,w,h=bound
        if( min_height<=h < max_height and mini_width<=w <max_width):
            val_rec.append(bound)
            if len(val_rec)==4:
                break
    print(val_rec)
    x1,y1,w1,h1=val_rec[0]
    x2,y2,w1,h1=val_rec[1]
    x3,y3,w1,h1=val_rec[2]
    x4,y4,w1,h1=val_rec[3]
    # print(x1,y1)
    midx1=math.floor(x1+w1/2)
    midy1=math.floor(y1+h1/2)
    midx2=math.floor(x2+w1/2)
    midy2=math.floor(y2+h1/2)
    midx3=math.floor(x3+w1/2)
    midy3=math.floor(y3+h1/2)
    midx4=math.floor(x4+w1/2)
    midy4=math.floor(y4+h1/2)

    x_l=min(midx1,midx2,midx3,midx4)#l
    y_b=min(midy1,midy2,midy3,midy4)#b
    x_r=max(midx1,midx2,midx3,midx4)#r
    y_t=max(midy1,midy2,midy3,midy4)#t
    tl = (x_l , y_t )
    bl = (x_l , y_b)
    tr = (x_r, y_t)
    br = (x_r, y_b)
    t = max(tl[1], tr[1])
    b = min(bl[1], br[1])
    l = max(tl[0], bl[0])
    r = min(tr[0], br[0])
    assert l<r
    assert t>b
    src_corners = [tl, tr, bl, br]

   

    h_tot=y_t-y_b
    w_tot=x_r-x_l
    print("xmin",x_l)
    print("ymin",y_b)
    print("xmax",x_r)
    print("ymax",y_t)
    plt.axis('off')
    # show the image
   
   
    # shearing applied to x-axis
    M = np.float32([[1, 0.1, 0],
             	   [0.04, 1  , 0],
            	   [0, 0  , 1]])
# shearing applied to y-axis
    N = np.float32([[1,   0, 0],
             	  [0.5, 1, 0],
             	  [0,   0, 1]])
    # 662
    unwarped_roi = [(l, t), (r, t), (l, b), (r, b)]
    print(unwarped_roi)
    #mat_per=cv2.getPerspectiveTransform()
# apply a perspective transformation to the image                
    #sheared_img = cv2.warpPerspective(img,M,(int(cols*1.3),int(rows*1.3))) #ok

    mat_sheared_img =cv2.getPerspectiveTransform(np.float32(src_corners), np.float32(unwarped_roi))
    dewarped_size = img.shape[:2][::-1]
    sheared_img= cv2.warpPerspective(img, mat_sheared_img,dewarped_size)
    #print(sheared_img)

    resize=cv2.resize(sheared_img,(900,800))
    print("resize=",resize)
    imgshow("e",resize)#
    # disable x & y axis
    plt.axis('off')
    crop_img = sheared_img[ y_b:math.floor(y_b*1.2)+math.floor(h_tot*1.2),x_l:math.floor(x_l*1.2)+math.floor(w_tot*1.2)].copy()
    imgshow("cropped",cv2.resize( crop_img,((900,800))))
    #testvert= BBox
    img_re= cv2.resize(mask,(900,800))
    imgshow("image",img_re)
    #cv2.imwrite("green.png", green)



#crop_shear(img)
 


new_width = 500 # Resize
img_h,img_w,_ = img.shape
scale = new_width / img_w
img_w = int(img_w * scale)
img_h = int(img_h * scale)
img = cv2.resize(img, (img_w,img_h), interpolation = cv2.INTER_AREA)
img_orig = img.copy()
imgshow('Original Image (Resized)', img_orig)
#locate_markers(img)
#dewarp_image(img)
# Bilateral Filter
bilateral_filtered_image = cv2.bilateralFilter(img, 15, 190, 190) 
imgshow('Bilateral Filter', bilateral_filtered_image)

# Outline Edges
edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 150) 
imgshow('Edge Detection', edge_detected_image)

# Find Circles
contours, hierarchy = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # Edges to contours

contour_list = []
rect_list = []
position_list = []

for contour in contours:
    approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True) # Contour Polygons
    area = cv2.contourArea(contour)
    
    rect = cv2.boundingRect(contour) # Polygon bounding rectangles
    x_rect,y_rect,w_rect,h_rect = rect
    x_rect +=  w_rect/2
    y_rect += h_rect/2
    area_rect = w_rect*h_rect
    
    if ((len(approx) > 8) & (len(approx) < 23) & (area > 250) & (area_rect < (img_w*img_h)/5)) & (w_rect in range(h_rect-10,h_rect+10)): # Circle conditions
        contour_list.append(contour)
        position_list.append((x_rect,y_rect))
        rect_list.append(rect)

img_circle_contours = img_orig.copy()
cv2.drawContours(img_circle_contours, contour_list,  -1, (0,255,0), thickness=1) # Display Circles
for rect in rect_list:
    x,y,w,h = rect
    cv2.rectangle(img_circle_contours,(x,y),(x+w,y+h),(0,0,255),1)

imgshow('Circles Detected',img_circle_contours)

# Interpolate Grid
rows, cols = (6,7)
mean_w = sum([rect[2] for r in rect_list]) / len(rect_list)
mean_h = sum([rect[3] for r in rect_list]) / len(rect_list)
position_list.sort(key = lambda x:x[0])
max_x = int(position_list[-1][0])
min_x = int(position_list[0][0])
position_list.sort(key = lambda x:x[1])
max_y = int(position_list[-1][1])
min_y = int(position_list[0][1])
grid_width = max_x - min_x
grid_height = max_y - min_y
col_spacing = int(grid_width / (cols-1))
row_spacing = int(grid_height / (rows - 1))

# Find Colour Masks
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Convert to HSV space


    # lower boundary RED color range values; Hue (0 - 10)
#lower1 = np.array([0, 100, 20])
#upper1 = np.array([10, 255, 255])
    # upper boundary RED color range values; Hue (160 - 180)
#lower2 = np.array([160,100,20])
#upper2 = np.array([179,255,255])
#lower_mask = cv2.inRange(img_hsv, lower1, upper1)
#upper_mask = cv2.inRange(img_hsv, lower2, upper2)
 
#mask_red = lower_mask + upper_mask
#img_red = cv2.bitwise_and(img, img, mask=mask_red)

lower_red = np.array([130, 130, 100])  # Lower range for red colour space
upper_red = np.array([255, 255, 255])  # Upper range for red colour space

lower_red = np.array([160,50,50])
upper_red = np.array([180,255,255]) 
mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
img_red = cv2.bitwise_and(img, img, mask=mask_red)
imgshow("Red Mask",img_red)



lower_yellow = np.array([25, 50, 70])
upper_yellow = np.array([35, 255, 255])
mask_yellow = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
img_yellow = cv2.bitwise_and(img, img, mask=mask_yellow)
imgshow("Yellow Mask",img_yellow)

# Identify Colours
grid = np.zeros((rows,cols))
id_red = 1
id_yellow = -1
img_grid_overlay = img_orig.copy()
img_grid = np.zeros([img_h,img_w,3], dtype=np.uint8)

for x_i in range(0,cols):
    x = int(min_x + x_i * col_spacing)
    for y_i in range(0,rows):
        y = int(min_y + y_i * row_spacing)
        r = int((mean_h + mean_w)/5)
        img_grid_circle = np.zeros((img_h, img_w))
        cv2.circle(img_grid_circle, (x,y), r, (255,255,255),thickness=-1)
        img_res_red = cv2.bitwise_and(img_grid_circle, img_grid_circle, mask=mask_red)
        img_grid_circle = np.zeros((img_h, img_w))
        cv2.circle(img_grid_circle, (x,y), r, (255,255,255),thickness=-1)
        img_res_yellow = cv2.bitwise_and(img_grid_circle, img_grid_circle,mask=mask_yellow)
        cv2.circle(img_grid_overlay, (x,y), r, (0,255,0),thickness=1)
        if img_res_red.any() != 0:
            grid[y_i][x_i] = id_red
            cv2.circle(img_grid, (x,y), r, (0,0,255),thickness=-1)
        elif img_res_yellow.any() != 0 :
            grid[y_i][x_i] = id_yellow
            cv2.circle(img_grid, (x,y), r, (0,255,255),thickness=-1)
    
print('Grid Detected:\n', grid)
#imgshow('Img Grid Overlay',img_grid_overlay)
imgshow('Img Grid',img_grid)

# Generate Best Move
num_red = sum([np.count_nonzero(row == 1) for row in grid])
num_yellow = sum([np.count_nonzero(row == -1) for row in grid])

if not any([0 in row for row in grid]):
    grid_full = True
    print("Grid Full")
    
elif num_yellow < num_red:
    move = (Puissance4_intel.bestMove(grid*(-1), 1, -1), id_yellow)
    print("Yellow To Move: Column {}".format(move[0]+1))
    
else:
    move = (Puissance4_intel.bestMove(grid, 1, -1), id_red)
    print("Red To Move: Column {}".format(move[0]+1))
    

# Display Output
if any([0 in row for row in grid]):
    img_output = img_orig.copy()
    empty_slots = sum([ row[move[0]] == 0 for row in grid ])
    x = int(min_x + move[0] * col_spacing)
    y = int(min_y) + (empty_slots-1) * row_spacing
    if move[1] == id_red:
        cv2.circle(img_output, (x,y), r+4, (0,0,255),thickness=-1)
    if move[1] == id_yellow:
        cv2.circle(img_output, (x,y), r+4, (0,255,255),thickness=-1)
    cv2.circle(img_output, (x,y), r+5, (0,255,0),thickness=2)

   # imgshow("Output",img_output)

#locate_markers(img)
#imgshow("image crop",img)
  #  h=100
 #   w=200
   # crop_img = img_output[y:y+h, x:x+w]
   # imgshow("cropped", crop_img)
cv2.destroyAllWindows()

