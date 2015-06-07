# -----------------------------------------------------------------------
#                         Mark Frasca - 7 May 2015
# -----------------------------------------------------------------------
# Routine to track Joe in real time and identify safe landing zones
# Included central area detection for safe landing zone
# To do: Select best landing spots
#
#

# Import OpenCV & Numpy
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Get Camera feed
camera_feed = cv2.VideoCapture(0)

# Return live camera image size in pixels
framesize_x=camera_feed.get(3)
framesize_y=camera_feed.get(4)

# Define the codec and create VideoWriter object
fourcc = cv2.cv.CV_FOURCC(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

### Load image data for testing
##print framesize_x,framesize_y

while(1):

    # Live camera feed
    _,frame = camera_feed.read()
    # Load image data for testing
##    frame = cv2.imread('image.jpg',1)
##    framesize_x, framesize_y = frame.shape[:2]

    #Convert the current frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# ------------------------------------------------------------------------------------------------
#                       Landing site analysis,
# ------------------------------------------------------------------------------------------------

    # Convert to grey scale for analysis and equalise image
    gray_image1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image1)
##    cv2.imshow('gray1',gray_image1)

    # Landing site - texture filters
##    blur = cv2.blur(gray_image,(3,3))
##    blur=cv2.medianBlur(gray_image,5)
    blur = cv2.bilateralFilter(gray_image,9,75,75)
    cv2.imshow('gray image',blur)

    # Use thresholding to identify clearings
    ret,thresh1 = cv2.threshold(blur,150,255,cv2.THRESH_BINARY)
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        # Increase number of iterations to reduce size and freqency of landing zones
    mask1 = cv2.erode(thresh1,element1, iterations=10)
##    cv2.imshow('mask -A',mask1)
    mask1 = cv2.dilate(mask1,element1,iterations=6)
##    cv2.imshow('mask -B',mask1)
##    mask1 = cv2.erode(mask1,element1)
    cv2.imshow('mask -C',mask1)


    #Create Contours for all safe zones & cycle through contours to identify safe zones
    contours, hierarchy = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        # Draw rectangle around safe zones
        cv2.rectangle(frame, (x,y),(x+w,y+h), (255,0,0), 3)
        # Find area of contour
        area=cv2.contourArea(contour)
        step=20 # pixel step across image
        if area>10000:
            for i in range(x,x+w-step,step):
                delta_ij=20 # delta step in pixels wide & high
                for j in range(y,y+h-step,step):
                    if mask1[j+delta_ij,i+delta_ij]>0:
                        cv2.circle(frame,(i+delta_ij,j+delta_ij),2,(0,255,0),-1)

        # finding centroids of contour and draw a circle there. These are the designed landing coordinates
        M = cv2.moments(contour)
        cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        if mask1[cy,cx]>0:
            cv2.circle(frame,(cx,cy),4,(0,255,0),-1)

# ------------------------------------------------------------------------------------------------

    #Define the threshold for finding a blue object with hsv
    lower_blue = np.array([100,0,0])
    upper_blue = np.array([179,255,255])
##    lower_blue = np.array([100,0,0])
##    upper_blue = np.array([130,245,120])

    #Create a binary image, where anything blue appears white and everything else is black
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #Get rid of background noise using erosion and fill in the holes using dilation and erode the final image on last time
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.erode(mask,element, iterations=2)
    mask = cv2.dilate(mask,element,iterations=2)
    mask = cv2.erode(mask,element)


    #Create Contours for all blue objects & identify largest blue object in image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maximumArea = 0
    bestContour = None

    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > maximumArea:
            bestContour = contour
            maximumArea = currentArea

     #Create a bounding box around the biggest blue object which is Joe's pants
    if bestContour is not None:
        x,y,w,h = cv2.boundingRect(bestContour)
        # Draw rectangle around Joe
        cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255), 3)

        # Find centre of box region
        x_centre=(2*x+w)/2
        y_centre=(2*y+h)/2

        # Print Joe's label above bounding box
        cv2.putText(frame, 'Joe',(x-25,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, cv2.CV_AA)
        # Print Joes coordiantes in frame space
        cv2.putText(frame, str(x)+","+str(y),(x_centre,y_centre), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, cv2.CV_AA)

        #Stearing commands to lock onto Joe, allowing a tolerance of +/-20 pixels for locking on
        # X-direction
        if (framesize_x/2-x_centre)>20:
##            print framesize_x/2,x_centre
            cv2.putText(frame, 'Roll:Right',(25,25), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, cv2.CV_AA)
        elif (framesize_x/2-x_centre)<20:
            cv2.putText(frame, 'Roll:Left',(25,25), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, cv2.CV_AA)
        else:
            cv2.putText(frame, 'Roll:Locked on Joe',(25,25), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, cv2.CV_AA)
        # Y-direction
        if (framesize_y/2-y_centre)>20:
            cv2.putText(frame, 'Pitch:Down',(25,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, cv2.CV_AA)
        elif (framesize_y/2-y_centre)<20:
            cv2.putText(frame, 'Pitch:Up',(25,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, cv2.CV_AA)
        else:
            cv2.putText(frame, 'Pitch:Locked on Joe',(25,50), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255), 1, cv2.CV_AA)

    #Show the original camera feed with a bounding box overlayed
##    res=[]
##    test = cv2.bitwise_and(frame,mask1,res=res)
    cv2.imshow('Frame',frame)
    out.write(frame)
##    cv2.imshow('Landing',mask1)

    #Show the contours in a seperate window
##    cv2.imshow('mask',mask)

    #Use this command to prevent freezes in the feed
    k = cv2.waitKey(5) & 0xFF

    #If escape is pressed close all windows
    if k == 27:
        break

cv2.destroyAllWindows()
