import Augmentor
import os
path = os.getcwd()
p = Augmentor.Pipeline(path)
p.rotate(probability=0.4, max_left_rotation=5, max_right_rotation=5)
p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=4)
p.sample(200)
