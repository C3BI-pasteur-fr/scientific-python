import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

def illustration_residuals(dataframe, fitted_model, response='Y', factor='Group'):
    _, axes = plt.subplots(1, 2)
    sns.boxplot(x=factor, y=response, data=dataframe, ax=axes[0])
    ax = axes[1]
    sns.boxplot(x=dataframe[factor], y=fitted_model.resid, ax=ax)
    ax.set_ylabel('residuals')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.axhline(0, linestyle='--', color='red', linewidth=1);

design_matrix_to_str = lambda g: repr(g.data.obj)

def side_by_side(*args, sep='    |    '):
    """
    Horizontally concatenate multiline string representations of objects.

    Example:

    >>> import numpy as np
    >>> rng = np.random.default_rng(seed=1)
    >>> a = rng.integers(10, size=(5,3))
    >>> b = rng.random(size=(6,1))
    >>> c = rng.choice(list('abcd'), size=(4,2))
    >>> print(side_by_side(str(a), str(b), str(c)))
    [[4 5 7]   |  [[0.54959369]   |  [['b' 'a']
     [9 0 1]   |   [0.02755911]   |   ['b' 'a']
     [8 9 2]   |   [0.75351311]   |   ['b' 'd']
     [3 8 4]   |   [0.53814331]   |   ['a' 'b']]
     [2 8 2]]  |   [0.32973172]   |
               |   [0.7884287 ]]  |
    >>>
    """
    strs = [ design_matrix_to_str(arg) for arg in args ]
    row_series = [ s.split('\n') for s in strs ]
    nrows = max([ len(rows) for rows in row_series ])
    padded_row_series = []
    for rows in row_series:
        max_len = max([len(row) for row in rows])
        padded_rows = []
        r = 0
        for r, row in enumerate(rows):
            padded_rows.append(row + ' '*(max_len-len(row)))
        empty_row = ' '*max_len
        r += 1
        while r < nrows:
            padded_rows.append(empty_row)
            r += 1
        padded_row_series.append(padded_rows)
    concatenated_rows = [ sep.join(rows) for rows in zip(*padded_row_series) ]
    return '\n'.join(concatenated_rows)

def illustration_2way_data(data, response='height', factorA='sun', factorB='water'):
    sns.boxplot(data=data, x=factorA, y=response, hue=factorB)
    ax = sns.swarmplot(data=data, x=factorA, y=response, hue=factorB, dodge=True, palette='dark:k');
    ax.legend([], frameon=False);
    # redraw the legend
    import matplotlib
    colored_patches = [ child for child in ax.get_children() if isinstance(child, matplotlib.patches.Rectangle) ][:-1]
    ax.legend(colored_patches, [ patch.get_label() for patch in colored_patches ], title='water');

def interaction_plot(data, response='height', factorA='water', factorB='sun'):
    ax = sns.swarmplot(data=data, x=factorA, y=response, hue=factorB);

    # get the colors used by swarmplot, to tell interaction_plot which colors to use;
    # and get the objects drawn by swarmplot to redraw the legend after calling interaction_plot
    sun_levels = np.unique(data[factorB])
    colors = {}
    colored_points = []
    for child in ax.get_children():
        if child.get_label() in sun_levels:
            try:
                color = child.get_facecolor()
            except AttributeError:
                color = child.get_color()
            colors[child.get_label()] = color
            colored_points.append(child)

    # interaction plot
    from statsmodels.graphics.factorplots import interaction_plot
    colors = [ colors[sun] for sun in np.sort(sun_levels) ]
    interaction_plot(x=data[factorA], trace=data[factorB], response=data[response],
                     ax=ax, colors=colors, markers='d'*len(sun_levels), markerfacecolor='w');

    # redraw the legend to remove the duplicates from interaction_plot
    ax.legend(colored_points, [ points.get_label() for points in colored_points ], title=factorB);

def illustration_multiple_comparisons(power=0.8, type1_error_rate=0.05):
    true_grid = np.zeros((20, 60), dtype=bool)
    true_grid[:10,-10:] = True

    rejection_grid = np.array([[ np.random.rand() <= type1_error_rate for _ in range(60) ] for _ in range(20)])
    rejection_grid[:10,-10:] = [[ np.random.rand() <= power for _ in range(10)] for _ in range(10)]

    _, axes = plt.subplots(1, 2, figsize=(13.3,4.1))
    for ax, title, grid in zip(axes[::-1], ('true', 'observed (actual test results)'), (true_grid, rejection_grid)):
        ax.imshow(grid, cmap='seismic')
        ax.set_title(title)
        ax.axis("off");

def confidence_intervals(all_comparisons):
    y = 0
    post_hoc_tests = all_comparisons[['coef', 'Conf. Int. Low', 'Conf. Int. Upp.', 'reject-hs']]
    for y, contrast in enumerate(post_hoc_tests.index):
        mean, lower_bound, upper_bound, reject = post_hoc_tests.loc[contrast]
        plt.errorbar(mean, -y, lolims=True, xerr=[[mean-lower_bound], [upper_bound-mean]], yerr=0, linestyle='', c='red' if reject else 'black')
        plt.text(mean, -y, contrast, ha='center', va='top')
    plt.axvline(0, color='darkorange')
    plt.yticks([]);

def illustration_regression(patients, model, example_patient=173, response='Response', predictor='CHUK', scatter_label='Patient', line_label='Model prediction'):
    ax = sns.scatterplot(data=patients, x=predictor, y=response, label=scatter_label)
    sm.graphics.abline_plot(model_results=model, ax=ax, label=line_label)
    plt.legend()

    x = patients.loc[example_patient, predictor]
    expected_value = patients.loc[example_patient, response]
    predicted_value = model.fittedvalues[example_patient]
    ax.plot([x, x], [predicted_value, expected_value], 'r-', zorder=0)
    ax.plot(patients[predictor], model.fittedvalues, 'g+');

def illustration_regression_residuals(patients, model, example_patient=173, response='Response', predictor='CHUK', scatter_label='Patient', line_label='Model prediction'):
    ax = sns.scatterplot(x=predictor, y='residuals', label=scatter_label,
        data={predictor: patients[predictor], 'residuals': model.resid})
    ax.axhline(0, linestyle=':', lw=1, label=line_label)
    plt.legend()

    x = patients.loc[example_patient, predictor]
    example_patient_residual = model.resid[example_patient]
    ax.plot([x, x], [0, example_patient_residual], 'r-', zorder=0);

def illustration_outlier(x, y, high_leverage_point, cooks_distant_point):
    _, axes = plt.subplots(1, 2, figsize=(13.3,4.1))
    for ax, influential_point in zip(axes, set([high_leverage_point, cooks_distant_point])):
        sns.scatterplot(x=x, y=y, ax=ax)
        sns.regplot(x=x, y=y, ax=ax, scatter=False, label=f'{influential_point:d} included')
        selection = np.ones(len(x), dtype=bool)
        selection[influential_point] = False
        sns.regplot(x=x[selection], y=y[selection], scatter=False, ax=ax, label=f'{influential_point:d} excluded')
        xi, yi = x[influential_point], y[influential_point]
        ax.plot(xi, yi, 'r.', markersize=14)
        ax.text(xi, yi, f'{influential_point:d}')
        ax.legend()

def illustration_monotonous_functions():
    _, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax, tr in zip(axes.T, (lambda x: 1./x, np.log, lambda x: x*x)):
        x = np.linspace(1, 50, 30)
        y = tr(x)
        scale = y.max() - y.min()
        y += .05 * scale * stats.norm.rvs(size=x.size)
        x_grid = np.linspace(1, 50, 100)
        ax[0].plot(x, y, 'b+')
        ax[0].plot(x_grid, tr(x_grid), 'r-')
        ax[1].plot(tr(x), y, 'b+')
        ax[1].plot(tr(x_grid), tr(x_grid), 'r-')

def illustration_nonlinear_regression(df, y_th, model, order):
    x = df['x']
    x_grid = np.linspace(x.min(), x.max(), 100)
    if order == 2:
        X_grid = np.stack((np.ones_like(x_grid), x_grid, x_grid**2), axis=1)
    else:
        X_grid = np.stack([x_grid**p for p in range(order+1)], axis=1)
    # y_grid = model.predict(X_grid) # bug?
    beta = model.params
    y_grid = np.dot(X_grid, beta)
    ax = sns.scatterplot(x='x', y='y', data=df, label='observations')
    ax.plot(x, y_th, 'r-', label='true')
    ax.plot(x_grid, y_grid, 'g-', label='predicted')
    ax.legend();

# support functions

def poly(x, order):
    return np.stack([x**k for k in range(order+1)], axis=1)

def fit(x, y, order):
    return sm.OLS(y, poly(x, order)).fit()

def fit_models(x, y, order=6):
    return {k: fit(x, y, k) for k in range(1, order+1)}

def predict(model, x, order):
    X = poly(x, order)
    beta = model.params
    y_pred = np.dot(X, beta)
    return y_pred

def sum_of_squares(y_predicted, y_expected=None):
    y_ = np.mean(y_predicted) if y_expected is None else y_expected
    y_ = y_predicted - y_
    return np.dot(y_, y_)

def R2(y_predicted, y_expected):
    residual_ss = sum_of_squares(y_predicted, y_expected)
    total_ss = sum_of_squares(y_expected)
    return 1 - residual_ss / total_ss

def illustration_R2_poly(x, y, x_test, y_test, order=6):
    models = fit_models(x, y, order)

    R2_train_data = {}
    R2_test_data = {}
    for order in models:
        trained_model = models[order]
        y_pred = predict(trained_model, x_test, order)
        R2_train_data[order] = trained_model.rsquared
        R2_test_data[order] = R2(y_pred, y_test)

    ax = plt.gca()
    ax.plot(list(R2_train_data.keys()), list(R2_train_data.values()), 'b-', label='train')
    ax.plot(list(R2_test_data.keys()), list(R2_test_data.values()), 'g-', label='test')
    ax.set_xlabel('polynomial order')
    ax.set_ylabel('$R^2$')
    ax.axvline(3, color='r', linestyle=':', linewidth=1, label='true')
    ax.legend();

def illustration_AIC_BIC_poly(x, y, order=6):
    models = fit_models(x, y, order)

    AIC = {}
    BIC = {}
    for order in models:
        trained_model = models[order]
        AIC[order] = trained_model.aic
        BIC[order] = trained_model.bic

    ax = plt.gca()

    orders, criteria = list(AIC.keys()), list(AIC.values())
    ax.plot(orders, criteria, 'b-', label='AIC')
    k = np.argmin(criteria)
    ax.plot(orders[k], criteria[k], 'ro', markerfacecolor='none')
    ax.annotate('AIC best candidate',
        xy=(orders[k], criteria[k]),
        xytext=(orders[k]-.2, criteria[k]+6),
        arrowprops=dict(arrowstyle="->"),
    )

    orders, criteria = list(BIC.keys()), list(BIC.values())
    ax.plot(orders, criteria, 'g-', label='BIC')
    k = np.argmin(criteria)
    ax.plot(orders[k], criteria[k], 'ro', markerfacecolor='none')
    ax.annotate('BIC best candidate',
        xy=(orders[k], criteria[k]),
        xytext=(orders[k]-.6, criteria[k]+7),
        arrowprops=dict(arrowstyle="->"),
    )

    ax.set_xlabel('polynomial order')
    ax.axvline(3, color='r', linestyle=':', linewidth=1, label='true')
    ax.legend();

