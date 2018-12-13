import numpy as np
# a = [[[147, 196, 210],
#        [198, 214, 224],
#        [243, 244, 246]],
#        [[223, 210, 173],
#        [222, 209, 163],
#        [193, 160, 125]]]
# print(np.shape(a))

# slice1 = a[0:1:1]
# # print(slice1)

# slice2 = a[1:2]
# print(slice2)

# #,np.shape(a)

each_patch=  [[0.60892632, 0.59992741, 0.51893721],
 [0.64188382, 0.58033332, 0.50119696],
 [0.66268438, 0.59834609, 0.45036803],
 [1,2,3],
 [5,6,7]] 

each_match = [[[0.60892632, 0.59992741, 0.51893721],
       [0.64188382, 0.58033332, 0.50119696],
       [0.66268438, 0.59834609, 0.45036803]], [[0.63389631, 0.59694271, 0.49176709],
       [0.64210677, 0.60450592, 0.47145677],
       [0.68896064, 0.57115908, 0.44621803]]] 
matches = [[[[[0.60892632, 0.59992741, 0.51893721],
       [0.64188382, 0.58033332, 0.50119696],
       [0.66268438, 0.59834609, 0.45036803]], [[0.63389631, 0.59694271, 0.49176709],
       [0.64210677, 0.60450592, 0.47145677],
       [0.68896064, 0.57115908, 0.44621803]]], [[[0.63389631, 0.59694271, 0.49176709],
       [0.64210677, 0.60450592, 0.47145677],
       [0.68896064, 0.57115908, 0.44621803]], [[0.60892632, 0.59992741, 0.51893721],
       [0.64188382, 0.58033332, 0.50119696],
       [0.66268438, 0.59834609, 0.45036803]]]]]

# # print(each_match[0:1:1][0])
# match_location = []
# if np.array_equal(each_match[0:1:1][0],each_patch):
# 	match_location.append(each_location)
# 	print("match_location",match_location)

# print(each_location[:,1])
each_location =[63, 58,62,59,34,45,33,44]
# a = np.partition(each_location, 7, axis=-1, kind='introselect', order=None)
# print(a)

similarity = [[899, 1],[109,2],[200,3],[33,4]] 
copied_sim = similarity
            # print("np",np.amin(copied_sim,0)) 
sorted_sim = sorted(copied_sim)
# print("sorted",sorted_sim[0][0],sorted_sim[0][1],sorted_sim[1][0])
        # print(sorted_sim)
match = []
if sorted_sim[0][0] < 0.7 * sorted_sim[1][0]:
#         	# print(sorted_sim,sorted_sim[0][0],sorted_sim[1][0])
	match.append([[78,45],each_patch[sorted_sim[0][1]]])
	print(match)
	# print("1",similarity,"2",copied_sim, "3",sorted_sim,"4",similarity,)
#         	# print("match",match)
#         	# location.append([each_location1,new_corners2[sorted_sim[0][1]]])
#         	locations_A.append(each_location1)
#         	locations_B.append(new_corners2[sorted_sim[0][1]])
