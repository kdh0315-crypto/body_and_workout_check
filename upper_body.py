import math

#FHA

def calculate_fha(neck_center, head, width, height):
    # neck pixcel 
    neck_x = neck_center[0] * width
    neck_y = neck_center[1] * height

    # ear middle point pixcel
    head_x = head[0] * width
    head_y = head[1] * height

    # neck to ear horizontal distance
    dx = abs(head_x - neck_x) 

    # neck to ear vertical distance
    dy = neck_y - head_y  # save direction info

    # same point check 
    if dx < 1e-6 and abs(dy) < 1e-6:
        return None

    # connection line to horizontal line angle   
    fha_deg = math.degrees(math.atan2(dy, dx))

    # limit 0 to 90 degree
    return max(0.0, min(90.0, fha_deg))


#FSA

#shoulder tilt angle
#thoracic kyphosis angle