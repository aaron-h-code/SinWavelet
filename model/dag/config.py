''' rotation degree '''
rotations = [0, 90, 180, 270]
flipping = ['noflip', 'left-right', 'bottom-up', 'front-back']
cropping = ['nocrop', 'corner1', 'corner2', 'corner3', 'corner4']
augment_list = {
                 'rotation': rotations,
                 'flipping' : flipping,
                 'cropping' : cropping
               }
