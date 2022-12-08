from scipy.linalg import null_space
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca as gca
from matplotlib.pyplot import grid as grid

G = 6.673 * 10 ** (-11)
R = 6371000
M = 5.972 * 10 ** 24
h_atm = 2500000
sec_orb_vel = 11200
sat_velocity = 7700
time = 90 * 60
NP = np.array([0, 0, 1])
NP = NP / np.linalg.norm(NP)

img = plt.imread('blue_marble2.jpg')
theta = np.linspace(0, np.pi, img.shape[0])
phi = np.linspace(0, 2 * np.pi, img.shape[1])
count = 180
theta_inds = np.linspace(0, img.shape[0] - 1, count).round().astype(int)
phi_inds = np.linspace(0, img.shape[1] - 1, count).round().astype(int)
theta = theta[theta_inds]
phi = phi[phi_inds]
img = img[np.ix_(theta_inds, phi_inds)]

theta, phi = np.meshgrid(theta, phi)
x_sp = R * np.sin(theta) * np.cos(phi)
y_sp = R * np.sin(theta) * np.sin(phi)
z_sp = R * np.cos(theta)


def sat_tr(north_coord, west_coord, sat_h):
    sat_height = sat_h
    nc = north_coord * np.pi / 180
    wc = west_coord * np.pi / 180
    init_pos = np.array([np.cos(nc) * np.cos(wc),
                         np.cos(nc) * np.sin(wc), np.sin(nc)])
    init_pos = init_pos / np.linalg.norm(init_pos)
    ort_pl = null_space([init_pos])
    norm_orb_pl = np.cross(NP, init_pos)
    vel_v = ort_pl @ np.array([[-4], [3]])
    vel_v = vel_v / np.linalg.norm(vel_v)

    r0 = init_pos * (R + sat_height)
    v0 = vel_v * sat_velocity
    np_d = NP * (R + sat_height)

    tspan = np.linspace(0, 2 * time, 10 ** 5)

    x0 = np.hstack((r0, v0.T[0]))

    if sat_height > h_atm:
        a = "The satellite doesn't enter the Earth atmosphere"
    else:
        a = f'The satellite enters the Earth atmosphere (height - {sat_height / 1000} km)'

    if sec_orb_vel > sat_velocity:
        b = "The satellite doesn't leave the Earth orbit"
    else:
        b = "The satellite leaves the Earth orbit"

    m = f'{a}\n{b}'

    def funcs(x, t):
        r = x[0:3]
        v = x[3:6]
        drdt = v
        dvdt = (-G * M * r) / ((np.linalg.norm(r) ** 3))
        return np.concatenate((drdt, dvdt))

    x = odeint(funcs, x0, tspan)
    kinet_en = [0] * tspan
    pot_en = [0] * tspan

    for i in range(len(tspan)):
        kinet_en[i] = 0.5 * (x[i][3:6]) @ (x[i][3:6])
        pot_en[i] = -(0.5 * G * M) / np.linalg.norm((x[i][0:3]))
    return x, tspan, m, r0, np_d, kinet_en, pot_en


def show_vp(tspan, x):
    fig = plt.figure()
    ax_r = fig.add_subplot(1, 2, 1)
    ax_r.set_title('Position change')
    ax_r.set_xlabel('t axis (s)')
    ax_r.set_ylabel('r axis (m)')
    plt.plot(tspan, x[:, 0:3], color='g')
    grid(color='k', linestyle='--', linewidth=0.2)

    ax_v = fig.add_subplot(1, 2, 2)
    ax_v.set_title('Velocity change')
    ax_v.set_xlabel('t axis (s)')
    ax_v.set_ylabel('v axis (m/s)')
    plt.plot(tspan, x[:, 3:6], color='r')
    grid(color='k', linestyle='--', linewidth=0.2)


def total_energy(tspan, k_e, p_e):
    fig3 = plt.figure()
    ax_r = gca()
    ax_r.set_xlabel('t axis (s)')
    ax_r.set_ylabel('E axis (J)')
    plt.plot(tspan, k_e[:], color='g')
    plt.plot(tspan, p_e[:], color='steelblue')
    plt.plot(tspan, k_e[:] + p_e[:], color='orange')
    line1, = ax_r.plot([1, 2, 3], label='potential energy')
    line2, = ax_r.plot([1, 2, 3], label='total energy')
    line3, = ax_r.plot([1, 2, 3], label='kinetic energy')
    ax_r.legend(handles=[line1, line2, line3])
    grid(color='k', linestyle='--', linewidth=0.2)


def show_tr(x, m, r0, np_d):
    fig2 = plt.figure()
    ax = fig2.add_subplot(projection='3d')
    ax.set_title(m)
    ax.set_xlabel('X axis (m)')
    ax.set_ylabel('Y axis (m)')
    ax.set_zlabel('Z axis (m)')
    ax.plot3D(x[:, 0], x[:, 1], x[:, 2], color='r')
    ax.plot3D(r0[0], r0[1], r0[2], 'o', color='cyan')
    ax.plot3D(np_d[0], np_d[1], np_d[2], 'o', color='g')
    ax.plot_surface(x_sp.T, y_sp.T, z_sp.T, facecolors=img / 255, cstride=1, rstride=1)
    ax.axis('scaled')
    ax = gca()
    grid(color='k', linestyle='--', linewidth=0.2)


x, tspan, m, r0, np_d, kin_en, pot_en = sat_tr(90, 0, 408000)
# x2 = sat_tr(-77, -77, 408000)[0]

show_vp(tspan, x)
total_energy(tspan, kin_en, pot_en)
show_tr(x, m, r0, np_d)
# show_tr(x2, m, r0, np_d)
plt.show()
