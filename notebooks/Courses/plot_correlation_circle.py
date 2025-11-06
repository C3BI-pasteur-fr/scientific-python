def plot_correlation_circle(pca, feature_names, dims=(1, 2), ax=None):
    """
    Plot correlation circle for given PCA object.
    
    pca : fitted sklearn.decomposition.PCA
    feature_names : list of variable names (columns of X)
    dims : tuple of components to plot (1-based indices), e.g. (1, 2)
    ax : optional matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    d1, d2 = dims[0] - 1, dims[1] - 1   # convert to 0-based
    
    # Loadings (components) shape: (n_components, n_features)
    # Correlation between variables and components for standardized X:
    # corr = loadings * sqrt(eigenvalues)
    loadings = pca.components_
    eigvals = pca.explained_variance_
    
    xs = loadings[d1, :] * np.sqrt(eigvals[d1])
    ys = loadings[d2, :] * np.sqrt(eigvals[d2])

    # Draw arrows
    for i, (x, y) in enumerate(zip(xs, ys)):
        ax.arrow(0, 0, x, y,
                 head_width=0.02, head_length=0.02, length_includes_head=True)
        ax.text(x * 1.05, y * 1.05, feature_names[i],
                ha='center', va='center')

    # Unit circle
    circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='black', linestyle='--')
    ax.add_artist(circle)

    # Axes formatting
    ax.set_xlabel(f"PC{dims[0]}")
    ax.set_ylabel(f"PC{dims[1]}")
    ax.axhline(0, linewidth=0.5)
    ax.axvline(0, linewidth=0.5)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal', 'box')
    ax.set_title("Correlation circle")
    
    return ax
