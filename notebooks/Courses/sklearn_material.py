import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import decomposition
from scipy import stats

def illustration_pca_transform(X, origin, unit_axes, axes_norm):
    def arrow3d(name, origin, vec, color='red', relative_cone_height=.2):
        rod = np.stack(
            (origin, origin + (1 - relative_cone_height) * vec),
            axis=0,
        )
        traces = []
        traces.append(go.Scatter3d(
            name=name,
            x=rod[:,0],
            y=rod[:,1],
            z=rod[:,2],
            mode='lines',
            line=dict(color=color),
            showlegend=False,
        ))
        cone_tip = origin + vec
        cone_main_axis = relative_cone_height * vec
        traces.append(go.Cone(
            name=name,
            x=[cone_tip[0]],
            y=[cone_tip[1]],
            z=[cone_tip[2]],
            u=[cone_main_axis[0]],
            v=[cone_main_axis[1]],
            w=[cone_main_axis[2]],
            anchor='tip',
            sizeref=1,
            colorscale=[(0, color), (1, color)],
            showscale=False,
        ))
        return traces

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            name='observation',
            x=X[:,0],
            y=X[:,1],
            z=X[:,2],
            mode='markers',
            marker=dict(size=1.5, line=dict(width=1)),
            showlegend=False,
        ),
    )
    for axis_index, axis_name in enumerate(['first', 'second', 'third']):
        for trace in arrow3d(
            f'{axis_name} principal axis',
            origin,
            unit_axes[axis_index] * 5,
        ) + arrow3d(
            'xyz'[axis_index],
            np.zeros_like(origin),
            np.eye(3)[axis_index] * 5,
            'green',
        ):
            fig.add_trace(trace)

    fig.show()

def illustration_double_scatter3d(X, y, axis_names=('first PC', 'second PC', 'third PC'), labels=[('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]):
    fig = make_subplots(rows=1, cols=2, specs=[[{'is_3d': True}, {'is_3d': True}]])

    # with group labels
    for name, species_y in labels:
        species_data = X[y==species_y, :]
        fig.add_trace(
            go.Scatter3d(
                name=name,
                x=species_data[:,0],
                y=species_data[:,1],
                z=species_data[:,2],
                mode='markers',
                marker=dict(size=2, line=dict(width=1)),
            ),
            row=1,
            col=2,
        )

    # without group labels
    fig.add_trace(
        go.Scatter3d(
            name='All',
            x=X[:,0],
            y=X[:,1],
            z=X[:,2],
            mode='markers',
            marker=dict(size=2, line=dict(width=1)),
        ),
        row=1,
        col=1,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=axis_names[0],
            yaxis_title=axis_names[1],
            zaxis_title=axis_names[2],
        ),
    )

    for scene_index in (1, 2):
        fig.update_layout({
            f'scene{scene_index}': {axis+'axis_title': axis_name \
            for axis, axis_name in zip('xyz', axis_names)}})

    fig.show()

def illustration_4D_pca_weights(pca, Pdf):
    _, axes = plt.subplots(1, 4)
    common_kwargs = dict(size=10, orient="h", jitter=False, linewidth=1, edgecolor="w")
    for j in range(4):
        ax = axes[j]
        sns.stripplot(x=pca.components_[j,:], y=Pdf.columns, ax=ax, **common_kwargs)
        if j > 0:
            ax.set_yticklabels([])
        ax.yaxis.grid(True)
        ax.axvline(0, color='k', linestyle=':', linewidth=1)
        ax.set_ylabel(None)
        ax.set_xlabel(Pdf.index[j])

def scree_plot(X):
    pca = decomposition.PCA()
    pca.fit(X)

    ax_left = ax = plt.gca()
    ax_left.plot(range(1, X.shape[1]+1), pca.explained_variance_, 'b-')
    ax_left.set_ylabel('$\\lambda$', color='b')
    ax_left.tick_params(axis='y', colors='b')

    total_variance = np.sum(pca.explained_variance_)
    cumulated_explained_variance = np.cumsum(pca.explained_variance_)
    ax_right = ax.twinx()
    ax_right.plot(np.r_[0, cumulated_explained_variance / total_variance], 'g-')
    ax_right.set_ylabel('cumulated explained variance ratio', color='g')
    ax_right.tick_params(axis='y', colors='g')

    ax.set_xlabel('component');

def illustration_probabilistic_pca(Xdf):
    _, axes = plt.subplots(2, 2, figsize=(13.3,8.2))
    for axes_row, feature_name in zip(axes, ('sepal length (cm)', 'petal length (cm)')):
        for ax in axes_row:
            sns.scatterplot(x=feature_name, y='petal width (cm)', data=Xdf, ax=ax)
        X_ = Xdf[[feature_name, 'petal width (cm)']].values
        ax.plot(*X_.mean(axis=0), 'r+')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        step = min(xlim[1]-xlim[0], ylim[1]-ylim[0]) / 100
        x, y = np.arange(xlim[0], xlim[1], step), np.arange(ylim[0], ylim[1], step)
        x_grid, y_grid = np.meshgrid(x, y)
        grid = np.stack((x_grid.flatten(), y_grid.flatten()), axis=1)
        z = stats.multivariate_normal(np.mean(X_, axis=0), np.cov(X_.T)).pdf(grid)
        z_grid = z.reshape(x_grid.shape)
        ax.contour(x_grid, y_grid, z_grid, 4);

