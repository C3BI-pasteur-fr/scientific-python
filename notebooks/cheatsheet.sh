#!/bin/bash
# vim: noai:noet:ts=2:sw=2

valid_symbol="[[:alpha:]][[:alnum:]_]*"

declare -A namespaces
namespaces[np]=numpy;
namespaces[pd]=pandas
namespaces[plt]="matplotlib.pyplot"
namespaces[sns]=seaborn
namespaces[px]="plotly.express"
namespaces[widgets]="ipywidgets.widgets"
namespaces[io]="skimage.io"
namespaces[stats]="scipy.stats"
namespaces[sm]="statsmodels.api"
namespaces[smf]="statsmodels.formula.api"
namespaces[anova]="statsmodels.stats.anova"
namespaces[diagnostic]="statsmodels.stats.diagnostic"
namespaces[power]="statsmodels.stats.power"
namespaces[decomposition]="sklearn.decomposition"
namespaces[manifold]="sklearn.manifold"
namespaces[umap]=umap

namespace_order="np pd plt sns px widgets io stats sm smf anova diagnostic power decomposition manifold umap"

for namespace in $namespace_order; do
	echo "${namespaces[$namespace]}"
	echo "---------"

	query="[^[:alnum:]_]?$namespace\.($valid_symbol)\("

	cat *.ipynb | sed -nE "/$query/s/^.*$query.*$/\1/p" | sort | uniq

	echo
done

unset -v namespaces namespace_order

exclude="formulaic IPython pyplot widgets express pathlib skimage scipy datasets anova diagnostic power"

args=$(echo $exclude | tr ' ' '|')

echo others
echo "---------"
query="from ((\.?$valid_symbol)+) import (([, ]*$valid_symbol)+)"
cat *.ipynb | sed -nE "/$query/s/.*$query.*$/\1: \3/p" | sort | uniq | grep -Ev $args
