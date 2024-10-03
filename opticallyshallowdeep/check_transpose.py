# Always output row, column, band 
def check_transpose(img):
    #if the #of bands is greater than the number of x or y cords
    y,x,b=img.shape 
    if b>y or b>x:
        img=img.transpose(1,2,0)
    return img