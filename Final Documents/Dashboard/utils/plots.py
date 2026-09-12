"""Shared Plotly figure builders for the Movie Genome Dashboard."""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

CLUSTER_COLOURS = px.colors.qualitative.Plotly
OUTLIER_COLOURS = {
    'tier1': '#FFA500',
    'tier2': '#FF4500',
    'tier3': '#8B0000',
    'clustered': '#4CAF50',
}


def _get_cluster_colour(i: int) -> str:
    return CLUSTER_COLOURS[i % len(CLUSTER_COLOURS)]


def umap_scatter_2d(
    x, y, colour_values, hover_text,
    colour_label="Cluster", title="UMAP 2D Projection",
    colour_is_categorical=True
):
    """Interactive 2D UMAP scatter using Scattergl for performance."""
    fig = go.Figure()
    if colour_is_categorical:
        unique_vals = sorted(set(colour_values))
        for val in unique_vals:
            mask = [c == val for c in colour_values]
            label = str(val)
            fig.add_trace(go.Scattergl(
                x=np.array(x)[mask],
                y=np.array(y)[mask],
                mode='markers',
                name=label,
                marker=dict(size=4, opacity=0.7),
                text=np.array(hover_text)[mask],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))
    else:
        fig.add_trace(go.Scattergl(
            x=x, y=y, mode='markers',
            marker=dict(
                size=4, opacity=0.7,
                color=colour_values,
                colorscale='Turbo',
                colorbar=dict(title=colour_label),
            ),
            text=hover_text,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))
    fig.update_layout(
        title=title,
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        height=600, template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a",
    )
    return fig


def umap_scatter_3d(
    x, y, z, colour_values, hover_text,
    colour_label="Cluster", title="UMAP 3D Projection",
    colour_is_categorical=True
):
    """Interactive 3D UMAP scatter."""
    fig = go.Figure()
    if colour_is_categorical:
        unique_vals = sorted(set(colour_values))
        for val in unique_vals:
            mask = [c == val for c in colour_values]
            fig.add_trace(go.Scatter3d(
                x=np.array(x)[mask],
                y=np.array(y)[mask],
                z=np.array(z)[mask],
                mode='markers',
                name=str(val),
                marker=dict(size=3, opacity=0.6),
                text=np.array(hover_text)[mask],
                hovertemplate="<b>%{text}</b><extra></extra>",
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode='markers',
            marker=dict(
                size=3, opacity=0.6,
                color=colour_values,
                colorscale='Turbo',
                colorbar=dict(title=colour_label),
            ),
            text=hover_text,
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))
    fig.update_layout(
        title=title, height=700, template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a",
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
    )
    return fig


def feature_radar(values, names, title="Genome Profile", overlay_values=None, overlay_name=None, top_n=15):
    """Radar chart showing top N features by value."""
    idx = np.argsort(values)[-top_n:][::-1]
    sel_names = [names[i] for i in idx]
    sel_vals = [float(values[i]) for i in idx]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sel_vals, theta=sel_names, fill='toself', name=title,
    ))
    if overlay_values is not None:
        overlay_sel = [float(overlay_values[i]) for i in idx]
        fig.add_trace(go.Scatterpolar(
            r=overlay_sel, theta=sel_names, fill='toself',
            name=overlay_name or "Comparison", opacity=0.5,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, max(sel_vals) * 1.1])),
        title=title, height=500, template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a",
    )
    return fig


def mini_radar(feat_names, feat_vals, color='rgba(136,192,208,0.7)', fill_color='rgba(136,192,208,0.15)',
               height=180, width=200):
    """Compact radar chart for inline display next to cluster/sub-cluster titles.

    All features are used as axes (for consistent shape across charts), but
    only features with non-zero values display their label. Inactive axes
    show as blank spokes so the polygon shape stays comparable.

    Args:
        feat_names: list of feature label strings (shared across all charts)
        feat_vals:  list of corresponding float values (0 for inactive features)
        color:      line colour
        fill_color: fill colour
        height/width: pixel dimensions
    Returns:
        Plotly Figure configured for compact inline display.
    """
    # Build display labels: show name only for active (non-zero) features
    display_names = [n if v > 0 else '' for n, v in zip(feat_names, feat_vals)]

    # Close the polygon
    r = list(feat_vals) + [feat_vals[0]] if feat_vals else []
    theta = list(display_names) + [display_names[0]] if display_names else []

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r, theta=theta, fill='toself',
        fillcolor=fill_color,
        line=dict(color=color, width=1.5),
        marker=dict(size=3, color=color),
        # Show real feature name on hover even for zero-value axes
        customdata=list(feat_names) + [feat_names[0]] if feat_names else [],
        hovertemplate='%{customdata}: %{r:.3f}<extra></extra>',
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=False,
                range=[0, max(feat_vals) * 1.15] if feat_vals else [0, 1],
            ),
            angularaxis=dict(
                tickfont=dict(size=8, color='#aaa'),
                rotation=90,
                direction='clockwise',
            ),
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=10, b=10),
        height=height,
        width=width,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def sparsity_heatmap(matrix, feature_names, film_labels=None, sample_n=200):
    """Sparsity heatmap for Data Health page."""
    n = matrix.shape[0]
    if n > sample_n:
        idx = np.random.RandomState(42).choice(n, sample_n, replace=False)
        matrix = matrix[idx]
        if film_labels is not None:
            film_labels = [film_labels[i] for i in idx]

    fig = px.imshow(
        matrix,
        labels=dict(x="Feature", y="Film", color="Value"),
        x=feature_names,
        y=film_labels if film_labels else None,
        color_continuous_scale="Blues",
        aspect="auto",
    )
    fig.update_layout(
        title=f"Feature Sparsity Heatmap ({matrix.shape[0]} films sampled)",
        height=600, template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a",
    )
    return fig


def distribution_histogram(values, feature_name):
    """Histogram of a single feature's distribution."""
    fig = px.histogram(
        x=values, nbins=20,
        labels={"x": feature_name, "y": "Count"},
        title=f"Distribution: {feature_name}",
    )
    fig.update_layout(height=400, template="plotly_dark")
    return fig


def soft_membership_bar(probs, cluster_names_dict, top_n=5):
    """Horizontal bar chart of soft cluster memberships."""
    sorted_idx = np.argsort(probs)[::-1][:top_n]
    names = []
    vals = []
    for i in sorted_idx:
        name = cluster_names_dict.get(i, {}).get('name', f'Cluster {i}')
        names.append(name)
        vals.append(float(probs[i]))

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation='h',
        marker_color=[_get_cluster_colour(int(i)) for i in sorted_idx],
    ))
    fig.update_layout(
        title="Soft Cluster Membership",
        xaxis_title="Probability", yaxis_title="",
        height=300, template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def metric_line_chart(df, x_col, y_col, colour_col=None, title=""):
    """Line chart of a metric across experiment runs."""
    fig = px.line(
        df, x=x_col, y=y_col, color=colour_col,
        markers=True, title=title,
    )
    fig.update_layout(height=400, template="plotly_dark")
    return fig


def cluster_feature_bars(feature_deltas, feature_names_list, title="Discriminative Features", top_n=15):
    """Horizontal bar chart of discriminative features (delta from global mean)."""
    sorted_idx = np.argsort(np.abs(feature_deltas))[::-1][:top_n]
    names = [feature_names_list[i] for i in sorted_idx]
    deltas = [float(feature_deltas[i]) for i in sorted_idx]
    colours = ['#4CAF50' if d > 0 else '#F44336' for d in deltas]

    fig = go.Figure(go.Bar(
        x=deltas, y=names, orientation='h',
        marker_color=colours,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Delta from corpus mean",
        height=max(300, top_n * 25),
        template="plotly_dark", plot_bgcolor="#05000a", paper_bgcolor="#05000a",
        yaxis=dict(autorange="reversed"),
    )
    return fig
