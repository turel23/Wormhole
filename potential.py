import numpy as np
import matplotlib.pyplot as plt

b0 = 1.0
E = 1.0

r = np.linspace(b0 * 1.001, 6, 1000)

def veff(r, a):
    omega = 2 * a / r**3
    b = b0**2 / r
    # at photon sphere bc = 3/2 * sqrt(-6a) for a < 0, else use a reference bc
    if a < 0:
        r_ps = np.sqrt(-6 * a)
        bc = 1.5 * np.sqrt(-6 * a)
    else:
        bc = 3.0  # reference value for a >= 0
    L = bc * E
    term1 = (E - omega * L)**2
    term2 = L**2 / r**2
    return -(1 - b / r) * (term1 - term2)

a_values = [-0.8, -0.4, 0.0]
colors = ['darkblue', 'steelblue', 'black']
labels = ['$a = -0.4$', '$a = -0.2$', '$a = 0$']

fig, ax = plt.subplots(figsize=(7, 5))

for a, color, label in zip(a_values, colors, labels):
    V = veff(r, a)
    ax.plot(r, V, color=color, label=label, linewidth=1.8)

ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')

# mark photon sphere locations for a < 0
for a in [-0.8, -0.4]:
    r_ps = np.sqrt(-6 * a)
    ax.axvline(r_ps, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)

ax.set_xlabel('$r$', fontsize=13)
ax.set_ylabel('$V_\\mathrm{eff}(r)$', fontsize=13)
ax.set_title('Effective potential for rotating Ellis wormhole', fontsize=12)
ax.set_ylim(-2, 2)
ax.set_xlim(b0, 6)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('veff_plot.png', dpi=300, bbox_inches='tight')
plt.show()