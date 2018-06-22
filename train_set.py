#This file is used to obtain training angles between the snake and tager location
import random
import math

rangeL = 0.0
rangeU = 20.0
tar_location = (10.00,12.50)
for _ in range(50):
	dist = 0.0
	while (dist<1.5):
		i = random.uniform(rangeL,rangeU)
		j = random.uniform(rangeL,rangeU)
		rel_x = tar_location[0] - i
		rel_y = tar_location[1] - j
		dist = math.sqrt((rel_x)*(rel_x) + (rel_y)*(rel_y))
	alphaR = alphaL = 0
	if rel_x < 0:
		if rel_y > 0: 
			alphaR = 90
			alphaL = -180
		else:
			alphaR = 180
			alphaL = -90
	else:
		val = rel_y / dist
		alpha = math.asin(val)*(180/math.pi)
		if alpha > 0:
			alphaR = alpha
			alphaL = -180
		else:
			alphaR = 180
			alphaL = alpha					

