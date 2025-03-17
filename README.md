#### Broad Overview

This reposity contains the results of one primary evolutionary optimization algorithm, which ran over about 30 generations in total. Each generation in the evolution produces 3 robots by mutating upon the 'best' robot from the prior generation. The mutation has been updated from prior labs to include not only changing the size of the robots themselves, but also adding/removing different parts in the iterations. Additionally, there is an updated, much more effective version of the actuation optimization from lab 4. In effect, the actuation is now more intense when the robot is detected to be 'upright' and when laying on its side follows a more regularly periodic oscillatory motion, meant to emulate galloping. The pictures file contains a load of pictures and a few gifs from throughout the evolution algorithm.


### Shape Construction

In modification to the original diffmpm file from difftaichi, this reposity contains the capability to construct a robot based off of a single function call. The parameters to the function are as follows (included within the 'main' function of the primary file):

within optimization_params:

these are continuous variables meant to be changed incrementally over the course of the simulation
in order as they appear in the array:

rec_wid - base rectangle width
rec_height - base rectangle height
branch_wid or branch_rad - if the branched segment is a rectangle, width of the branch rectangle. If a half-circle, radius of the half-circle
branch_height - if branched segment is rectangle, height of the rectangle. otherwise ignored
branch_rel - relative location along the main rectangle segment to place the branched segment (eg. 0.5 -> in the middle, 0.25 -> closer to one side)
connector_wid - width of connecting rectangle between main segments
connector_height - height of connecting rectangle between main segments

within constant_params:

originally not changed throughout evolution, is now modified discretely instead.
in order as they appear:

num_segments - number of main rectangle segments
init_x - inital x starting location (set to 0)
init_y - initial y starting location (also set to 0)
branch_id - 0 for rectangular branches, 1 for half-circle ones
connectivity - defines direction branches go in. 0 for left, 1 for right, 2 for up


### Evolutions and Mutations

The evolution algorithm defines a set number of generations and robots within each generation. The cost function is optimized based on how far to the right the robot gets during the simulation, with the best robot from the prior generation serving as the initial one in the next generation.

The variance itself is produced as a result of the function mutate(), which randomly changes a pre-set number of the variables described above. For the continuous variables in optimization_params the algorithm will decide to randomly either completely change the base value, or use the prototype robot for the generation as a template, changing the value by an incremental amount. For the discrete variables, if one of them is selected as a variable to be mutated, the algorithm randomly picks an acceptable value (eg. connectivity can only be 0, 1, or 2, not 3).

The evolution algorithm was somewhat handicapped by the fact that my computer is reeeeeaaaallly on its last legs over here. I was continually running into memory issues and generally just slow compilation times, but I did checkpoint my method so that I could restart easily, which is saved within the restart.txt files in the repo. Due to the same problems I had to set a maximum number of particles the robot could adopt, retrying the mutation algorithm until it got an acceptable number. The algorithm also partly modifies the visualize() function to produce a series of pngs for the robot's run into the folder, which are all compiled within there. I think that if the number of robots per generation were increased, which I really tried to do but simply couldn't get it to compile consistently (even over night) it would be easier to end with more variability in the final robot. 


### Modifying Actuation

To modify the actuation behavior, I found myself struggling with but eventually learning the way that taichi kernels work and made sure to be consistent with my use of functions throughout. The basic idea is that while the robot is upright, I wanted a spring-like behavior with low frequency and phase to get it to leap onto its side. Then, while on its side, it shifts to a more oscillatory type motion that is trying to emulate galloping. I was able to eventually observe some of the galloping behavior by the end, but I think I was further limited by my processing power there since if I could increase the time of each simulation it would be easier to shift into this behavioral regime.

In terms of implementation, this included adding the compute_orientation function, which determines the relative position of the robot in relation to the 'ground', where if the angle between a rightward x vector and the current robot's vector is below 45 degrees, the robot is considered 'sideways' and switches to the galloping actuation mode. The output from this function is either a 0 or 1, which is then passed to the main compute_actuation function. This function now has two modes as well, deciding which calculations to use for the frequency and phase to adjust the weights on the actuation itself. 

Getting the kernels to work properly here was honestly the most difficult part of the project. I had to use ChatGPT a fair bit to try to understand what was going on between all the different functions and even ended up with an evolution algorithm that worked fine at one point with an actuation optimization that certainly did nothing. But I eventually got there and could see some of the differences in robot movement behavior in the pictures.

