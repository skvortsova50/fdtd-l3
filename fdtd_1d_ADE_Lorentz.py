import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def gaussian_pulse(x, mean, sigma):
    return np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))


class FDTD1D_Maxwell:
    """
    1D FDTD (Yee)
    """

    def __init__(self, x_dim, time_tot, c, dx, S,
                 eps=1.0, mu=1.0):

        self.x_dim = x_dim
        self.time_tot = time_tot
        self.c = c
        self.dx = dx
        self.S = S
        self.dt = S * dx / c

        # --- тепер eps — СКАЛЯР ---
        self.eps = eps
        self.mu = mu

        # зонди
        self.probe_ref = 60
        self.probe_trn = 170

        self.signal_ref = np.zeros(time_tot)
        self.signal_trn = np.zeros(time_tot)

        # поля
        self.E = np.zeros(x_dim)
        self.E_old = np.zeros(x_dim)
        self.J = np.zeros(x_dim)
        self.J_old = np.zeros(x_dim)
        self.H = np.zeros(x_dim - 1)

        self.t_step = 0

        self.source_pos = x_dim // 2
        self.sigma_0 = 5

        # --- шар ---
        self.layer_start = 110
        self.layer_end = 140
        self.eps_r = 1.0
        # self.eps_inf = 4.0
        # self.eps_s = 80.0
        # self.delta_eps = self.eps_s - self.eps_inf  # = 3.0
        # self.tau = 10.0  # або 10.0

        self.eps_inf = 2.0
        self.eps_s = 4.0
        self.omega0 = 2 * np.pi * 5
        self.delta = 0.05 * self.omega0
        self.delta_eps = self.eps_s - self.eps_inf  # = 3.0

        # self.delta2tau = self.dt / self.tau
        # self.k = (2 - self.delta2tau) / (2 + self.delta2tau)
        # self.beta = 2 * self.delta_eps * self.delta2tau / (2 + self.delta2tau)
        # self.C1 = 1
        # self.C2 = 2 * self.dt / (2 * self.eps_inf + self.beta)
        # self.C3 = (1 + self.k) * self.dt / (2 * self.eps_inf + self.beta)

        self.alpha = (2 - self.omega0 * self.dt ** 2) / (1 + self.delta * self.dt)
        self.ksi = (self.delta * self.dt - 1) / (1 + self.delta * self.dt)
        self.gamma = self.eps_inf * self.omega0 ** 2 * self.dt ** 2 / (1 + self.delta * self.dt)
        denominator = 2 * self.delta_eps + 0.5 * self.gamma
        self.C1 = 0.5 * self.gamma / denominator
        self.C2 = 2 * self.eps_inf / denominator
        self.C3 = 2 * self.dt / denominator

        # --- маска шару---
        self.mask = np.zeros(x_dim, dtype=bool)
        self.mask[self.layer_start:self.layer_end] = True
        self.mask_mid = self.mask[1:-1]


        # --- TFSF ---
        self.iL = x_dim // 4
        self.iR = self.iL + 5

        self.t0 = 30 * self.dt
        self.tw = 10 * self.dt

        self.probe_ref = 30
        self.probe_trn = 170

        self.signal_ref = np.zeros(time_tot)
        self.signal_trn = np.zeros(time_tot)




    def g(self, tau):
        return np.exp(-((tau - self.t0) / self.tw) ** 2)

    def src_E(self, t, z):
        return self.g(t - z / self.c)

    def src_H(self, t, z):
        return - self.g(t - z / self.c) / np.sqrt(self.mu / self.eps)

    # -------------------------------
    # Один крок
    # -------------------------------

    def update_H(self):
        self.H[:] += (self.dt / (self.mu * self.dx)) * \
                     (self.E[1:] - self.E[:-1])

    def apply_TFSF_H(self):
        t = self.t_step * self.dt
        zL = self.iL * self.dx

        self.H[self.iL - 1] -= (self.dt / (self.mu * self.dx)) * self.src_E(t, zL)

    def update_E(self):
        curl = (self.H[1:] - self.H[:-1]) / self.dx

        E_mid = self.E[1:-1]
        J_mid = self.J[1:-1]
        E_mid_old = self.E_old[1:-1]
        J_mid_old = self.J_old[1:-1]

        # --- вакуум ---
        E_mid[~self.mask_mid] += self.dt * curl[~self.mask_mid] / self.eps

        # --- діелектрик ---
        self.update_lorentz_ade(E_mid, J_mid, E_mid_old, J_mid_old, curl)
        # self.update_debye_plrc(E_mid, psi_mid, curl)
        # self.update_dielectric_const(E_mid, curl)

    # def update_debye_ade(self, E_mid, J_mid, curl):
    #     E_mid_old = E_mid
    #     # --- оновлення E ---
    #     E_mid[self.mask_mid] += (
    #             self.C2 * curl[self.mask_mid]
    #             - self.C3 * J_mid[self.mask_mid]
    #     )
    #
    #     # --- оновлення J ---
    #     J_mid[self.mask_mid] = (
    #             self.k * J_mid[self.mask_mid]
    #             + ( self.beta / self.dt ) * (E_mid[self.mask_mid] - E_mid_old[self.mask_mid])
    #     )

    def update_lorentz_ade(self, E_mid, J_mid, E_mid_prev_prev, J_mid_prev_prev, curl):
        E_mid_prev = E_mid
        J_mid_prev = J_mid

        # --- оновлення E ---
        E_mid[self.mask_mid] =  self.C1 * E_mid_prev_prev[self.mask_mid] + self.C2 * E_mid_prev[self.mask_mid]
        + self.C3 * ( curl[self.mask_mid] - 0.5 * ( (1 + self.alpha) * J_mid_prev[self.mask_mid]
                                                    + self.ksi * J_mid_prev_prev[self.mask_mid]))


        # --- оновлення J ---
        J_mid[self.mask_mid] = (self.alpha * J_mid_prev[self.mask_mid] + self.ksi *  J_mid_prev_prev[self.mask_mid] +
                                (self.gamma / (2 * self.dt)) * (E_mid[self.mask_mid] - E_mid_prev_prev[self.mask_mid]))

    def update_dielectric_const(self, E_mid, curl):
        E_mid[self.mask_mid] += self.dt * curl[self.mask_mid] / (self.eps * self.eps_r)

    def apply_TFSF_E(self):
        t = self.t_step * self.dt
        t_half = t + 0.5 * self.dt

        zLh = (self.iL - 0.5) * self.dx

        self.E[self.iL] -= (self.dt / (self.eps * self.dx)) * self.src_H(t_half, zLh)

    def apply_BC(self):
        self.E[0] = self.H[0]
        self.E[-1] = - self.H[-1]

    def record(self):
        if self.t_step < self.time_tot:
            self.signal_ref[self.t_step] = self.E[self.probe_ref]
            self.signal_trn[self.t_step] = self.E[self.probe_trn]

    def epsilon(self, omega):
        return self.eps_inf + (self.delta_eps * self.omega0 ** 2) / ( self.omega0 ** 2 + 2j * omega * self.delta - omega * omega)


    def spectrum(self):
        ref_spectrum = np.fft.fftshift(np.fft.fft(self.signal_ref))
        trn_spectrum = np.fft.fftshift(np.fft.fft(self.signal_trn))
        freq = np.fft.fftshift(np.fft.fftfreq(len(ref_spectrum), d=self.dt))
        pos = freq > 0
        omega = 2 * np.pi * freq[pos]

        plt.figure()

        plt.plot(omega, self.dt * np.abs(ref_spectrum[pos]), 'o', color='r', markersize=4,
                 label="reflected spectrum")
        plt.plot(omega, self.dt * np.abs(trn_spectrum[pos]), 'o', color='b', markersize=4,
                 label="transmitted spectrum")
        plt.plot(omega, self.spectrum(omega).real, color='g', label="transmitted spectrum")
        plt.legend()
        plt.show()

    def step(self):
        E_prev = self.E.copy()
        J_prev = self.J.copy()

        self.update_H()
        self.apply_TFSF_H()

        self.update_E()

        self.apply_TFSF_E()
        self.apply_BC()
        self.record()

        self.E_old[:] = E_prev
        self.J_old[:] = J_prev

        self.t_step += 1


    def plot_signals(self):
        t = np.arange(self.time_tot) * self.dt

        plt.figure()
        plt.plot(t, self.signal_ref, label="Reflected")
        plt.plot(t, self.signal_trn, label="Transmitted")

        plt.legend()
        plt.grid()
        plt.xlabel("t")
        plt.ylabel("E")
        plt.show()

    # -------------------------------
    # Анімація
    # -------------------------------
    def animate(self):

        fig, ax = plt.subplots()

        ax.axvspan(self.layer_start, self.layer_end,
                   color='gray', alpha=0.3, label="dielectric")

        ax.set_xlim(0, self.x_dim)
        ax.set_ylim(-1.2, 1.2)
        ax.set_title("1D FDTD (Maxwell): E та H")
        ax.axvline(self.iL, ls=':', color='k')
        ax.axvline(self.probe_ref, ls='--', color='blue')
        ax.axvline(self.probe_trn, ls='--', color='red')

        lineE, = ax.plot([], [], lw=2, label="E")
        lineH, = ax.plot([], [], lw=2, linestyle="--", label="H")
        ax.legend()

        def update(frame):
            self.step()
            lineE.set_data(np.arange(self.x_dim), self.E)
            lineH.set_data(np.arange(self.x_dim - 1) + 0.5, self.H)
            return lineE, lineH

        ani = animation.FuncAnimation(
            fig, update,
            frames=self.time_tot,
            interval=50,
            blit=True,
            repeat=False
        )

        plt.show()
        return ani


# -------------------------------
# запуск
# -------------------------------
sim = FDTD1D_Maxwell(
    x_dim=200,
    time_tot=1200,
    c=1.0,
    dx=1.0,
    S=1.0
)

sim.animate()
sim.plot_signals()
sim.spectrum()
