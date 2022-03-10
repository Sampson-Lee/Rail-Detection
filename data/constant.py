# row anchors are a series of pre-defined coordinates in image height to detect lanes
# the row anchors are defined according to the evaluation protocol of CULane and Tusimple
# since our method will resize the image to 288x800 for training, the row anchors are defined with the height of 288
# you can modify these row anchors according to your training image resolution
# start from 80 with step 5

raildb_row_anchor = [200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320,
                     330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430, 440, 450,
                     460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580,
                     590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710]
