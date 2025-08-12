import cv2
import numpy as np

cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.imshow("Test Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
