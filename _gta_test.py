import cv2
import matplotlib.pyplot as plt
import numpy as np
from grabscreen import grab_screen



def bluring(img):
    """Returns the edges of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to gray to reduce computation calculus
    blur = cv2.GaussianBlur(gray, (5,5), 0)  # Average Average Filter to smooth the image. Often 5x5

    return blur




while (True):
    frame = grab_screen(region=(0, 150, 800, 600))  #1.778

    cropped_image = cv2.resize(frame, (214, 120))  #old 160
    # cropped_image = bluring(cropped_image)                

    # cv2.imshow('window', cropped_image)
    cv2.imshow('window2', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    if cv2.waitKey(25) & 0xff == ord('p'):
        cv2.destroyAllWindows()
        break

