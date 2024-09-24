import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from itertools import combinations

def plot_fluidvel_and_ratio(spinner_dfs, 
                            ratio_dfs,
                            xlims_fluidvel,
                            xlims_ratio,
                            ylims,
                            xlabels,
                            labels,
                            subtitles,
                            ylabel='Depth [m]',
                            colormap='Paired',
                            colorstartshift=0):
    """
    Plot fluid velocity (spinner) data against depth from a stage or completion test on axe 0 and 
    the ratio of these fluid velocity data on axe 1.

    Assumes that the first list of labels of `labels` in order correspond to the ordered spinner 
    data sets in `spinner_dfs` and the second list of labels correspond to those in `ratio_dfs`.
    """

    fig, ax = plt.subplots(1, 2, figsize=(7,9), sharey=True)
    ax[0].invert_yaxis()
    ax[0].set_ylim(ylims[0], ylims[1])
    ax[0].set_ylabel(ylabel)
    
    cmap = plt.cm.get_cmap(colormap)
    ax[0].set_xlim(xlims_fluidvel[0], xlims_fluidvel[1])
    ax[1].set_xlim(xlims_ratio[0], xlims_ratio[1])
            
    for i, title in enumerate(subtitles):
        ax[i].set_title(title)
    
    for i, xlabel in enumerate(xlabels):
        ax[i].set_xlabel(xlabel)
    
    for i, spin_df in enumerate(spinner_dfs):
        ax[0].plot(spin_df["FluidVel"], spin_df["Depth"], 
                    label=labels[0][i], color=cmap(i + colorstartshift))
        
    for i, ratio_df in enumerate(ratio_dfs):
        ax[1].plot(ratio_df["Ratio"], ratio_df["Depth"], 
                    label=labels[1][i], 
                    color=cmap(i + colorstartshift + len(spinner_dfs)))
    
    for i in range(0,2):
        ax[i].legend(frameon=True, loc='upper right')

    return fig, ax

def plot_spinner(spinner_dfs,     
                    xlims_spinner,
                    ylims,
                    xlabels,
                    labels,
                    subtitles,
                    ylabel='Depth [m]',
                    colormap='Paired'):
    """
    Plot spinner frequencey data against depth from a stage or completion test on axe 0

    Assumes that the list of labels of `labels` in order correspond to the ordered spinner data 
    sets in `spinner_dfs`.
    """

    fig, ax = plt.subplots(1, 1, figsize=(8,10), sharey=True)
    ax.invert_yaxis()
    ax.set_ylim(ylims[0], ylims[1])
    ax.set_ylabel(ylabel)
    
    cmap = plt.cm.get_cmap(colormap)
    ax.set_xlim(xlims_spinner[0], xlims_spinner[1])

    for i, title in enumerate(subtitles):
        ax.set_title(title)
    
    for i, xlabel in enumerate(xlabels):
        ax.set_xlabel(xlabel)
    
    for i, spin_df in enumerate(spinner_dfs):
        speeds = spin_df['Speed_Group'].unique()
        for speed in speeds:
            speed_spin_df = spin_df[spin_df['Speed_Group'] == speed]
            ax.plot(speed_spin_df["Frequency"], speed_spin_df["Depth"],
                        label=labels[0][i] + ', tool speed:' + speed, color=cmap(i))
    
    ax.legend(frameon=True, loc='upper right')

    return fig, ax

def add_casing_to_plot(ax, production_shoe, top_of_liner, terminal_depth, 
                       ax_index=0, include_liner=True):
    """
    Add the casing information to the `ax_index` contained in the list of axes, `ax`.
    """

    # blank well casing
    ax[ax_index].plot([-0.05, -0.05],[0, production_shoe],
        color = 'k', linewidth = 3, linestyle = 'solid', label = 'Casing')

    # perforated well casing
    if include_liner:
        ax[ax_index].plot([0., 0],[top_of_liner, terminal_depth],     
            color = 'k', linewidth = 2.5, linestyle = 'dashed', label = 'Liner')

    ax[ax_index].legend(loc='upper right')
    return None

def add_feedzone_to_plot(ax, start_depth, end_depth):
    """
    Adds feedzone to every plot axis in `ax` between `start_depth` and `end_depth` as a grey box
    """

    for ax_i in ax:
        ax_i.axhspan(ymin=start_depth, ymax=end_depth, color = 'grey', alpha=.2)
    return None

def add_feedzones_to_plot(ax, feedzones):
    """
    Adds `feedzones` to every plot axis in `ax` between `start_depth` and `end_depth` as a grey box 
    using `add_feedzone_to_plot`

    `feedzones` is a list of lists containing the start and end depths for each zone, [[980, 990], 
    [1123, 1128]].
    """

    for feedzone in feedzones:
        add_feedzone_to_plot(ax, feedzone[0], feedzone[1])
    return None

def add_feedzone_labels_to_plot(ax, feedzones, labels):
    """
    Adds a label to each shaded feed zone location created using `add_feedzones_to_plot` to every 
    plot axis in `ax`.

    `feedzones` is a list of lists containing the start and end depths for each zone, [[980, 990], 
    [1123, 1128]].

    `labels` is a list of strings, with equal length to `feedzones`, which are the label for each 
    zone.
    """

    if len(feedzones) != len(labels):
        raise IndexError("the length of feedzones must be the same as the length of labels")

    for i, feedzone in enumerate(feedzones):
        for ax_i in ax:
            ax_i.text(ax_i.get_xlim()[1], np.average(feedzone), labels[i], va="center", ha="center", size=11, 
                      bbox=dict(boxstyle="square", lw=1, fc="white", ec="black"))
    return None

def add_ratio_to_plot(ax, ratio_df, start_depth, end_depth):
    """
    Adds ratio to the second plot axis in `ax` between `start_depth` and `end_depth` as the mean of 
    the ratio values between these depths in ratio_df
    """
    
    ratio = ratio_df.loc[
        ratio_df["Depth"].apply(lambda x: x >= start_depth and x <= end_depth), "Ratio"].mean()
    ax[1].vlines(ratio, ymin=start_depth, ymax=end_depth, color = 'black', 
                 alpha = 1.0, lw=3, zorder=3) # zorder makes it plot on top of other lines
        
    return None

def compute_depth_offset_lsq(depths_baseline,
                            fluidvelocity_baseline,
                            depths_comparison,
                            spl_spin_comparison,
                            mean_vel_comparison,
                            search_range,
                            step_size):
    """
    Given a normalised baseline fluid velocity (its mean has been subtracted from it), evaluate and 
    return the best depth offset for the compared fluid velocity profile (which we normalise) in 
    the `search_range` using a least squares objective. 
    """
    
    offsets = np.arange(search_range[0], search_range[1]+step_size, step_size)
    objs = np.zeros(offsets.size)

    for i, offset in enumerate(offsets):
        depths_comparison = depths_baseline + offset
        objs[i] = np.sum(np.square((spl_spin_comparison(depths_comparison) - mean_vel_comparison) - 
                                   fluidvelocity_baseline))
        
    best_offset = round(offsets[np.argmin(objs)], 1)

    return best_offset

def compute_depth_offset_variance(depths_baseline,
                            fluidvelocity_baseline,
                            depths_comparison,
                            spl_spin_comparison,
                            search_range,
                            step_size):
    """
    Given a baseline fluid velocity evaluate and return the best depth offset for the compared 
    fluid velocity profile in the `search_range` by minimising the variance of the ratio between 
    the two profiles. 
    """
    
    offsets = np.arange(search_range[0], search_range[1]+step_size, step_size)
    objs = np.zeros(offsets.size)

    for i, offset in enumerate(offsets):
        depths_comparison = depths_baseline + offset
        objs[i] = np.var(spl_spin_comparison(depths_comparison) / fluidvelocity_baseline)
        
    best_offset = round(offsets[np.argmin(objs)], 1)

    return best_offset

def compute_spinner_interpolations_and_offsets(spin_baseline, 
                            spin_comparisons,
                            search_range=[-5,5],
                            comparison_range=None,
                            comparison_type='lsq', # can also be 'variance'
                            step_size=0.1,
                            min_depth=None,
                            max_depth=None):
    """
    `spin_comparisons` is required to be a list of spinner dataframes with depths sorted in 
    ascending order of magnitude (i.e. index 0 is the shallowest depth and index -1 is the deepest 
    depth).

    When `comparison_type` = 'lsq':
    - The offset is calculated by computing the least squares difference between a baseline 
    normalised cubic spline and comparison spline for a set of offsets in the `search_range` (-5 to 
    +5 metres) using `compute_depth_offset_lsq`. Normalised means that each profile has had its 
    mean magnitude subtracted from it, so that differences in magnitude don't dominate the 
    objective function. The extremities of both profiles related to the search range are ignored to 
    enable a fair calculation of the objective, e.g. for `search_range=[-5,5]` we ignore the first 
    and last 5 metres of data in the objective. If `comparison_range` is `None` then we make this 
    comparison across the entire data range. Otherwise we only consider depth values within the 
    comparison range.

    When `comparison_type` = 'variance':
    - The offset is calculated by minimising the variance of the ratio computed from the baseline 
    and comparison spline for a set of offsets in the `search_range` (default -5 to +5 metres) 
    using `compute_depth_offset_variance`. This should done for a `comparison_range` that is in a 
    region with no feed zones. The extremities of both profiles related to the search range are 
    ignored to enable a fair calculation of the objective, e.g. for `search_range=[-5,5]` we ignore 
    the first and last 5 metres of data in the objective. If `comparison_range` is `None` then we 
    make this comparison across the entire data range. Otherwise we only consider depth values 
    within the comparison range. This is a slight update on the method used by Grant, Wilson & 
    Bixley (2006).

    We recommend computing the offset relative to the first flow rate to mitigate for wireline 
    stretch over the course of the stage/completion test.

    `step_size` is used for both the step size for computing the interpolated fluid velocity 
    profiles at and the step size used for the depth offset within `search_range`. 

    Returns `baseline_df`, `comparison_dfs` and `best_offsets` where the `comparison_dfs` relate to 
    the newly offset and interpolated versions of spin_comparisons. As a result the depths 
    contained in `baseline_df` and each of the `comparison_dfs` may not be the same - this is done 
    to make sure any data in each of the profiles at the end of the ranges is not cut off when 
    visualising the plots. The function for calculating the ratios will compensate for this 
    (`compute_ratio_dfs`).
    """

    if type(spin_comparisons) is not list:
        raise TypeError("spin_comparisons must be a list. If you have only one spinner dataset, enclose it in brackets []")
    
    if comparison_type not in ['lsq', 'variance']:
        raise ValueError("comparison type must be either 'lsq' or 'variance'")
    
    if step_size < 0.099:
        raise ValueError("values of step_size lower than 0.1 are not reasonable")
    
    # make depths considered the smallest range of any collected data sets 
    # this shouldn't be too dissimilar given they will be from the same PTS run
    if min_depth is None:
        min_depth = np.ceil(max(spin_baseline["Depth"][0], 
                                max([spinner["Depth"][0] for spinner in spin_comparisons])))
    if max_depth is None:
        max_depth = np.floor(min(spin_baseline["Depth"].iloc[-1], 
                                min([spinner["Depth"].iloc[-1] for spinner in spin_comparisons])))

    if comparison_range is None:
        comparison_range = [min_depth, max_depth]
    else:
        comparison_range = [max(min_depth, comparison_range[0]), 
                            min(max_depth, comparison_range[1])]

    depths = np.arange(min_depth, max_depth+step_size, step_size)
    depths_baseline = np.arange(comparison_range[0]-search_range[0], 
                                comparison_range[-1]-search_range[1]+step_size, step_size)
    depths_comparison = np.zeros(depths_baseline.size)

    spl_spin_baseline = CubicSpline(spin_baseline["Depth"], spin_baseline["FluidVel"])
    spl_spin_comparisons = [CubicSpline(spinner["Depth"], spinner["FluidVel"]) 
                            for _, spinner in enumerate(spin_comparisons)]

    fluidvelocity_baseline = spl_spin_baseline(depths_baseline)
    best_offsets = np.zeros(len(spl_spin_comparisons))

    if comparison_type == 'lsq':
        # normalise splines by the mean fluid velocity of each profile
        mean_vel_base = np.mean(spin_baseline["FluidVel"])
        mean_vel_comps = [np.mean(spinner["FluidVel"]) for spinner in spin_comparisons]
        fluidvelocity_baseline_norm = fluidvelocity_baseline - mean_vel_base

        for i, spl_spinner in enumerate(spl_spin_comparisons):
            best_offsets[i] = compute_depth_offset_lsq(depths_baseline, fluidvelocity_baseline_norm,
                                            depths_comparison, spl_spinner, mean_vel_comps[i],
                                            search_range, step_size)
        
    elif comparison_type == 'variance':
        for i, spl_spinner in enumerate(spl_spin_comparisons):
            best_offsets[i] = compute_depth_offset_variance(depths_baseline, fluidvelocity_baseline,
                                            depths_comparison, spl_spinner, 
                                            search_range, step_size)

    # make depths considered the largest range of any collected data sets to the closest integer
    # this shouldn't be too dissimilar given they will be from the same PTS run
    min_depth = np.ceil(min(spin_baseline["Depth"][0], 
                                min([spinner["Depth"][0] - best_offsets[i] 
                                        for i, spinner in enumerate(spin_comparisons)])))

    max_depth = np.floor(max(spin_baseline["Depth"].iloc[-1], 
                                max([spinner["Depth"].iloc[-1] - best_offsets[i] 
                                        for i, spinner in enumerate(spin_comparisons)])))

    # depths we will return interpolants at
    depths = np.arange(min_depth, max_depth+step_size, step_size)

    # find first index greater than or equal to the respective depth
    ind1 = np.searchsorted(depths, spin_baseline["Depth"][0], side='left')
    ind2 = np.searchsorted(depths, spin_baseline["Depth"].iloc[-1], side='left') - 1

    depths_baseline = np.arange(depths[ind1], depths[ind2]+step_size, step_size)
    baseline_df = pd.DataFrame({"Depth": depths_baseline, 
                                "FluidVel":  spl_spin_baseline(depths_baseline)})

    comparison_dfs = []
    for i, best_offset in enumerate(best_offsets):
        ind1 = np.searchsorted(depths, spin_comparisons[i]["Depth"][0]-best_offset, side='left')
        ind2 = np.searchsorted(depths, spin_comparisons[i]["Depth"].iloc[-1]-best_offset, 
                               side='left') - 1

        depths_comparison = np.arange(depths[ind1], depths[ind2]+step_size, step_size)
        comparison_dfs.append(pd.DataFrame(
                            {"Depth": depths_comparison, 
                            "FluidVel": spl_spin_comparisons[i](depths_comparison+best_offset)}))

    return baseline_df, comparison_dfs, best_offsets

def compute_ratio_dfs(spinner_dfs):
    """
    `spinner_dfs` is a list of spinner dataframes each with fluid velocities recorded at the 
    overlapping depths with the same values. Assumes that the spinner dataframes are in order of 
    increasing injecting fluid velocity.
    """

    if len(spinner_dfs) < 2:
        raise TypeError("spinner_dfs must be a list containing dataframes from at least 2 fluid velocities")
    
    dfs = []
    for combo in combinations(np.arange(0, len(spinner_dfs), 1), 2):
        df = pd.merge_asof(spinner_dfs[combo[0]], spinner_dfs[combo[1]], 
                            on ='Depth', tolerance=0.001)
        df.dropna(inplace=True)
        df["Ratio"] = df["FluidVel_y"] / df["FluidVel_x"]
        dfs.append(df)
    
    return dfs