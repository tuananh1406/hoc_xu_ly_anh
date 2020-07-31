import cv2


def sap_xep(duong_bao, sap_xep="left-to-right"):
    '''
    Hàm sắp xếp các đường bao theo thứ tự tùy chỉnh
    '''
    # khởi tạo giá trị reverse mặc định và chỉ số i
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if sap_xep == "right-to-left" or sap_xep == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if sap_xep == "top-to-bottom" or sap_xep == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in duong_bao]
    (duong_bao, boundingBoxes) = zip(*sorted(
        zip(duong_bao, boundingBoxes),
        key=lambda b: b[1][i],
        reverse=reverse,
        ))
    # return the list of sorted contours and bounding boxes
    return (duong_bao, boundingBoxes)
