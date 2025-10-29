import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

"""
Plot a normal distribution with shaded area to illustrate probability mass and
confidence interval.

Parameters
----------
m : real
    sample mean
s : real
    standard error of the mean
alpha : real
    1 - probability mass

"""
def illustration_confidence_interval(m=46, s=1, alpha=0.05):
    b = 3
    grid = np.linspace(m-b*s, m+b*s, 200) # possible population mean values
    pdf = stats.norm(m, s).pdf
    prob = pdf(grid) # probability for the population mean

    plt.plot(grid, prob, 'r-', zorder=3)
    plt.axhline(0, color='k', linestyle=':', linewidth=1)
    plt.xlabel('population mean')
    plt.ylabel('probability density')
    plt.axvline(m, color='g', linestyle=':', linewidth=1, label='sample mean')

    u = stats.norm().isf(alpha / 2)
    ci_low = m - u * s
    ci_high = m + u * s

    plt.fill_between(grid, np.zeros_like(prob), prob, where=(ci_low<=grid)&(grid<=ci_high), alpha=.1)
    plt.plot([ci_low]*2, [0, pdf(ci_low)], color='b')#, label='confidence lower bound')
    plt.plot([ci_high]*2, [0, pdf(ci_high)], color='b')#, label='confidence upper bound')

    ml = (grid[0]+4*ci_low)/5
    pl = (2*pdf(ci_low)+pdf(m))/3
    plt.annotate(f'${alpha/2*100}\\%$',
        [ml, .1*pdf(ml)], [ml, pl],
        arrowprops=dict(arrowstyle="->"),
        horizontalalignment='center')

    ml1 = (4*m+ci_high)/5
    ml2 = (m+ci_high)/2
    plt.annotate(f'${(1-alpha)*100:.0f}\\%$ prob. mass',
        [ml1, pl], [ml2, (pdf(ml2)+pdf(m))/2],
        arrowprops=dict(arrowstyle="->"))

    line_width, head_length, height = pdf(m)/30, b*s/10, .5*pdf(ci_low)
    t = plt.arrow(ci_low+head_length, height, ci_high-ci_low-2*head_length, 0,
        width=line_width, head_length=head_length, linestyle='none')
    t = plt.arrow(ci_high-head_length, height, ci_low-ci_high+2*head_length, 0,
        width=line_width, head_length=head_length, linestyle='none')
    plt.text(m, height+line_width, f'${(1-alpha)*100:.0f}\\%$ confidence interval',
        ha='center')

    plt.legend(loc='upper left')

    plt.xlim([grid[0], grid[-1]])

def illustration_onesided_probabilitymass(z, N = stats.norm(0, 1), onesided_pvalue=None):
    if onesided_pvalue is None:
        onesided_pvalue = N.sf(z)
    grid = np.linspace(N.ppf(.001), N.ppf(.999), 100)
    pdf_curve, = plt.plot(grid, N.pdf(grid), 'r-')
    z_line, = plt.plot([z, z], [0, N.pdf(z)], '-', zorder=1)
    tail = grid[z<=grid]
    plt.fill_between(tail, np.zeros_like(tail), N.pdf(tail), alpha=.2)
    plt.axvline(0, linestyle='--', color='grey', linewidth=1)
    plt.axhline(0, linestyle='--', color='grey', linewidth=1)
    plt.xlim(grid[[0,-1]])
    plt.xlabel('$X$')
    plt.ylabel('probability density')
    plt.legend([pdf_curve, z_line], [r'$\mathcal{N}(0,1)$', '$z$'])
    plt.annotate(f'$\\approx {onesided_pvalue:.2f}$', (1.8, .03), xytext=(2, .13), arrowprops=dict(arrowstyle="->"))

def illustration_t_pdfs():
    grid = np.linspace(-3.1, 3.1, 100)

    dfs = [1, 2, 5, 20]

    for df, color in zip(
        dfs,
        ['blue', 'green', 'orange', 'red'],
    ):
        t = stats.t(df)
        plt.plot(grid, t.pdf(grid), '-', color=color)

    plt.axvline(0, linestyle='--', color='grey', linewidth=1)
    plt.axhline(0, linestyle='--', color='grey', linewidth=1)
    plt.xlim(grid[[0,-1]])
    plt.xlabel('$t$')
    plt.ylabel('probability density')
    plt.legend([ f'$df={df}$' for df in dfs ]);

def illustration_cohen_d():
    def plot_pdfs(cohen_d):
        grid = np.linspace(-3, 3+cohen_d, 100)
        x1 = stats.norm(0, 1).pdf(grid)
        x2 = stats.norm(cohen_d, 1).pdf(grid)
        plt.fill_between(grid, x1, alpha=.5)
        plt.fill_between(grid, x2, alpha=.5)
        plt.show()
    slider = widgets.FloatSlider(.5, min=0, max=4, step=.1)
    return interact(plot_pdfs, cohen_d=slider)

def illustration_skewness_kurtosis():
    skewed_dist = lambda sigma, x: np.exp( -.5*(np.log(x)/sigma)**2 ) / ( x*sigma*np.sqrt(2*np.pi) )
    heavy_tailed_dist = lambda scale, x: stats.cauchy.pdf(x, 0, scale)

    colors = ['blue', 'green', 'orange', 'red']
    _, axes = plt.subplots(1, 2, figsize=(13.3,4.1))

    grid = np.linspace(0, 3, 100)
    grid = grid[1:]

    ax = axes[0]
    for sigma, color in zip((.25, .5, 1), colors):
        ax.plot(grid, skewed_dist(sigma, grid), '-', color=color, label=f'$\\sigma={sigma:.2f}$')

    ax.axhline(0, linestyle='--', color='grey', linewidth=1)
    ax.set_xlim(grid[[0,-1]])
    ax.set_title('skewness')

    grid = np.linspace(-4, 4, 100)

    ax = axes[1]
    for s, color in zip((.5, 1, 2), colors):
        ax.plot(grid, heavy_tailed_dist(s, grid), '-', color=color, label=f'$scale={s:.2f}$')

    ax.axvline(0, linestyle='--', color='grey', linewidth=1)
    ax.axhline(0, linestyle='--', color='grey', linewidth=1)
    ax.set_xlim(grid[[0,-1]])
    ax.set_title('kurtosis')

    for ax in axes:
        ax.set_ylabel('probability density')
        ax.legend();

def illustration_chi2():
    grid = np.linspace(0, 20, 200)

    dfs = [2, 3, 5, 9]

    _, axes = plt.subplots(1, 2, figsize=(13.3,4.1))

    ax = axes[0]
    for df, color in zip(
        dfs,
        ['blue', 'green', 'orange', 'red'],
    ):
        chi2 = stats.chi2.pdf(grid, df)
        ax.plot(grid, chi2, '-', color=color)

    ax.axhline(0, linestyle='--', color='grey', linewidth=1)
    ax.set_xlim(grid[0],grid[-1])
    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('probability density')
    ax.legend([ f'$df={df}$' for df in dfs ])

    ax = axes[1]
    df, color = 2, 'blue'
    chi2 = stats.chi2.pdf(grid, df)
    ax.plot(grid, chi2, '-', color=color)
    ax.axhline(0, linestyle='--', color='grey', linewidth=1)
    ax.set_xlim(grid[0],grid[-1])
    ax.set_xlabel(r'$\chi^2$')
    ax.set_ylabel('probability density');

    A = [85, 86, 88, 75, 78, 94, 98, 79, 71, 80]
    B = [91, 92, 93, 85, 87, 84, 82, 88, 95, 96]
    C = [79, 78, 88, 94, 92, 85, 83, 85, 82, 81]
    bartlett_statistic, bartlett_pvalue = stats.bartlett(A, B, C)
    bartlett_statistic_line, = ax.plot([bartlett_statistic]*2, [0, stats.chi2.pdf(bartlett_statistic, df)], '-', zorder=1)

    tail = grid[bartlett_statistic<=grid]
    ax.fill_between(tail, np.zeros_like(tail), stats.chi2.pdf(tail, df), alpha=.2)

    ax.annotate(f'$\\approx {bartlett_pvalue:.2f}$', (4, .02), xytext=(8, .1), arrowprops=dict(arrowstyle="->"));

def illustration_falsealarms(type1_error_rate, power):
    true_grid = np.zeros((20, 60), dtype=bool)
    true_grid[:10,-10:] = True

    rejection_grid = np.array([[ np.random.rand() <= type1_error_rate for _ in range(60) ] for _ in range(20)])
    rejection_grid[:10,-10:] = [[ np.random.rand() <= power for _ in range(10)] for _ in range(10)]

    _, axes = plt.subplots(1, 2, figsize=(13.3,4.1))
    for ax, title, grid in zip(axes[::-1], ('true', 'observed (actual test results)'), (true_grid, rejection_grid)):
        ax.imshow(grid, cmap='seismic')
        ax.set_title(title)
        ax.axis("off")

