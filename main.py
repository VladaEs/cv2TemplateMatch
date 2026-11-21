import cv2

img = cv2.imread("images/source.jpg", 0) # 0 ->load as grayscale
ball = cv2.imread("images/ball.png", 0) # 0 -> load as grayscale

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
for method in methods: 
    img2 = img.copy()
    result = cv2.matchTemplate(img2, ball, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
        h, w = ball.shape
    cv2.rectangle(img2, location, (location[0] + w, location[1] + h), 255, 5)
    cv2.imshow(str(method), img2)
    cv2.waitKey(0)
cv2.destroyAllWindows()