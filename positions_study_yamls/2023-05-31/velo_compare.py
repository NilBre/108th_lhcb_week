import numpy as np
import matplotlib.pyplot as plt

'''
retest:
v1:
VPLeft: !<!alignment>
  position: [-0.16685116531259747 * mm, 0.07581057303501407 * mm, -0.32476858790721724 * mm]
  rotation: [0.00013042489865930677 * rad, -6.619201389037521e-05 * rad, 0.00278045686954009 * rad]
VPRight: !<!alignment>
  position: [0.1663322897829051 * mm, -0.0772498063280982 * mm, 0.3248126555818226 * mm]
  rotation: [-0.00012924619338004436 * rad, 6.717820038293502e-05 * rad, -0.0027805035182178976 * rad]

v2:
VPLeft: !<!alignment>
  position: [-0.1668409699518422 * mm, 0.07591248466657662 * mm, -0.32827424746264505 * mm]
  rotation: [0.00013041632929937765 * rad, -6.927403503933075e-05 * rad, 0.002780457673647003 * rad]
VPRight: !<!alignment>
  position: [0.16632678406065943 * mm, -0.07735267168707369 * mm, 0.32831637218580384 * mm]
  rotation: [-0.00012925476306256006 * rad, 7.02602215332393e-05 * rad, -0.0027805027216980934 * rad]

v3:
VPLeft: !<!alignment>
  position: [-0.1668499816640979 * mm, 0.07592967662939291 * mm, -0.3279776304191575 * mm]
  rotation: [0.00013041434663861162 * rad, -6.998711215657827e-05 * rad, 0.002780457859762439 * rad]
VPRight: !<!alignment>
  position: [0.16633709066855748 * mm, -0.07736973286059232 * mm, 0.32801935421008843 * mm]
  rotation: [-0.0001292567458057287 * rad, 7.097329865082423e-05 * rad, -0.00278050253748178 * rad]

v4:
VPLeft: !<!alignment>
  position: [-0.166841980047706 * mm, 0.07600559602653686 * mm, -0.3303004182139204 * mm]
  rotation: [0.00013039684843361792 * rad, -7.628045457312157e-05 * rad, 0.0027804595019690096 * rad]
VPRight: !<!alignment>
  position: [0.16633888518257348 * mm, -0.07744627219792101 * mm, 0.33033841404895353 * mm]
  rotation: [-0.00012927424472000248 * rad, 7.726664107022048e-05 * rad, -0.0027805009112837746 * rad]
'''

# velo only
velo_left_pos = [-0.16674936782654887, 0.07258445117058115, -0.3284784888130428] # mm
velo_right_pos = [0.16631836965950933, -0.07402496407650368, 0.32848004413704707] # mm

velo_left_rot = [0.00013022568678090047, -0.00013783997782589743, 0.002780475555815851] # rad
velo_right_rot = [-0.00012944541403099065, 0.00013882616434737847, -0.002780484994610483] # rad

# v1
v1_left_pos  = [-0.16685116531259747, 0.07581057303501407, -0.32476858790721724]
v1_right_pos = [0.1663322897829051, -0.0772498063280982, 0.3248126555818226]

v1_left_rot  = [0.00013042489865930677, -6.619201389037521e-05, 0.00278045686954009]
v1_right_rot = [-0.00012924619338004436, 6.717820038293502e-05, -0.0027805035182178976]
# v2
v2_left_pos  = [-0.1668409699518422, 0.07591248466657662, -0.32827424746264505]
v2_right_pos = [0.16632678406065943, -0.07735267168707369, 0.32831637218580384]

v2_left_rot  = [0.00013041632929937765, -6.927403503933075e-05, 0.002780457673647003]
v2_right_rot = [-0.00012925476306256006, 7.02602215332393e-05, -0.0027805027216980934]
# v3
v3_left_pos  = [-0.1668499816640979, 0.07592967662939291, -0.3279776304191575]
v3_right_pos = [0.16633709066855748, -0.07736973286059232, 0.32801935421008843]

v3_left_rot  = [0.00013041434663861162, -6.998711215657827e-05, 0.002780457859762439]
v3_right_rot = [-0.0001292567458057287, 7.097329865082423e-05, -0.00278050253748178]
# v4
v4_left_pos  = [-0.166841980047706, 0.07600559602653686, -0.3303004182139204]
v4_right_pos = [0.16633888518257348, -0.07744627219792101, 0.33033841404895353]

v4_left_rot  = [0.00013039684843361792, -7.628045457312157e-05, 0.0027804595019690096]
v4_right_rot = [-0.00012927424472000248, 7.726664107022048e-05, -0.0027805009112837746]

v_l_x = [velo_left_pos[0], v1_left_pos[0], v2_left_pos[0], v3_left_pos[0], v4_left_pos[0]]
v_r_x = [velo_right_pos[0], v1_right_pos[0], v2_right_pos[0], v3_right_pos[0], v4_right_pos[0]]
v_l_y = [velo_left_pos[1], v1_left_pos[1], v2_left_pos[1], v3_left_pos[1], v4_left_pos[1]]
v_r_y = [velo_right_pos[1], v1_right_pos[1], v2_right_pos[1], v3_right_pos[1], v4_right_pos[1]]
lenx = np.linspace(0, len(v_l_x), len(v_l_x))

plt.plot(lenx, v_l_x, 'kx')
plt.title('velo constant left half in x')
plt.xlabel('version')
plt.ylabel('velo half x constant [mm]')
plt.xticks(lenx, ['velo only', 'v1', 'v1_1', 'v1_2', 'v3'])
plt.show()
plt.clf()
plt.plot(lenx, v_r_x, 'kx')
plt.title('velo constant right half in x')
plt.xlabel('version')
plt.ylabel('velo half x constant [mm]')
plt.xticks(lenx, ['velo only', 'v1', 'v1_1', 'v1_2', 'v3'])
plt.show()
plt.clf()
plt.plot(lenx, v_l_y, 'kx')
plt.title('velo constant left half in y')
plt.xlabel('version')
plt.ylabel('velo half y constant [mm]')
plt.xticks(lenx, ['velo only', 'v1', 'v1_1', 'v1_2', 'v3'])
plt.show()
plt.clf()
plt.plot(lenx, v_r_y, 'kx')
plt.title('velo constant right half in y')
plt.xlabel('version')
plt.ylabel('velo half y constant [mm]')
plt.xticks(lenx, ['velo only', 'v1', 'v1_1', 'v1_2', 'v3'])
# plt.show()
plt.clf()


# x position vs y position, velo_only
plt.plot(velo_left_pos[0], velo_left_pos[1], 'b.', label='VELO only, left half')
plt.plot(velo_right_pos[0], velo_right_pos[1], 'bx', label='VELO only, right half')

plt.plot(v1_left_pos[0], v1_left_pos[1], 'r.', label='v1, left half')
plt.plot(v1_right_pos[0], v1_right_pos[1], 'rx', label='v1, right half')

plt.plot(v2_left_pos[0], v2_left_pos[1], 'k.', label='v2 left')
plt.plot(v2_right_pos[0], v2_right_pos[1], 'kx', label='v2 right')

plt.plot(v3_left_pos[0], v3_left_pos[1], 'g.', label='v3 left')
plt.plot(v3_right_pos[0], v3_right_pos[1], 'gx', label='v3 right')

plt.plot(v4_left_pos[0], v4_left_pos[1], 'y.', label='v4 left')
plt.plot(v4_right_pos[0], v4_right_pos[1], 'yx', label='v4 right')

# plt.plot(v3_left_pos[0], v3_left_pos[1], 'g.', label='v3 left')
# plt.plot(v3_right_pos[0], v3_right_pos[1], 'gx', label='v3 left')
print('y diff on left side in [mm]:', abs(velo_left_pos[1] - v1_left_pos[1]))
print('y diff on right side in [mm]:', abs(velo_right_pos[1] - v1_right_pos[1]))

plt.legend()

plt.xlabel('velo x [mm]')
plt.ylabel('velo y [mm]')
plt.title('velo x vs y')
plt.savefig('velo_constants.pdf')
plt.clf()

# x = [0, 1, 2]
# z_left = [velo_left_pos[2], v1_left_pos[2], v3_left_pos[2]]
# z_right = [velo_right_pos[2], v1_right_pos[2], v3_right_pos[2]]
#
# plt.plot(z_left, x, 'k.', label='z left')
# plt.plot(z_right, x, 'r.', label='z right')
# plt.legend()
# plt.xlabel('z pos')
# plt.ylabel('file number')
# plt.yticks(x, ['velo only', 'v1', 'v3'])
# plt.title('z positions for different runs')
# plt.show()
# plt.clf()
