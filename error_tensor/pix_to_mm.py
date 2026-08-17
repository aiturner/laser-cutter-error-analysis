import math 


# top left
x1 =  405.10    
y1 = 492.04

# top right
x2 = 4403.71
y2 = 484.42 

# bottom left
x3 = 412.5      
y3 = 4474.00

# bottom right
x4 = 4393.33    
y4 = 4468.70

xdiff = (x2+x4)/2 - (x1+x3)/2  
ydiff = (y3+y4)/2 - (y1+y2)/2  
diagonaldiff = math.sqrt((x1 - x4)**2 + (y1-y4)**2)

print(xdiff)
print(ydiff)
print (diagonaldiff/ math.sqrt(2))

print(f"{(xdiff + ydiff) / 20} pixels per mm")