__author__ = 'User'
import cv2


def outline_rect(image, rect, color):
    if rect is None:
        return
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x+w, y+h), color)


def copy_rect(src, dst, src_rect, dst_rect, interpolation=cv2.INTER_LINEAR):
    """Copy part of the source to part of the destination."""
    x0, y0, w0, h0 = src_rect
    x1, y1, w1, h1 = dst_rect
    dst[y1:y1+h1, x1:x1+w1] = cv2.resize(src[y0:y0+h0, x0:x0+w0], (w1, h1), interpolation = interpolation)


def swap_rect(src, dst, rect, interpolation=cv2.INTER_LINEAR):
    if dst is not src:
        dst[:] = src
    num_rect = len(rect)
    if num_rect < 2:
        return
    x, y, w, h = rect[num_rect - 1]
    temp = src[y:y+h, x:x+w].copy()
    i = num_rect - 2
    while i >= 0:
        copy_rect(src, dst, rect[i], rect[i+1], interpolation)
        i -= 1
        copy_rect(temp, dst, (0,0,w,h), rect[0], interpolation)
