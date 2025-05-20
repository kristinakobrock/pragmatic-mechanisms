
import numpy as np
from matplotlib import pyplot as plt
import itertools
from utils.concept_reps import retrieve_messages_freq_rank, retrieve_concepts_context
import torch
from utils.load_results import load_interaction
from language_analysis_local import MessageLengthHierarchical


def plot_heatmap(result_list,
                 mode,
                 plot_dims=(2, 2),
                 figsize=(7, 7),
                 ylims=(0.6, 1.0),
                 titles=('context-aware \ntrain', 'context-aware \nvalidation', 'context-unaware \ntrain', 'context-unaware \nvalidation'),
                 suptitle=None,
                 suptitle_position=1.03,
                 different_ylims=False,
                 n_runs=5,
                 matrix_indices=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
                 fontsize=18):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
    Allows for plotting multiple matrices according to plot_dims, and allows different modes:
    'max', 'min', mean', 'median', each across runs. """

    plt.figure(figsize=figsize)

    for i in range(np.prod(plot_dims)):

        if different_ylims:
            y_lim = ylims[i]
        else:
            y_lim = ylims

        heatmap = np.empty((3, 3))
        heatmap[:] = np.nan
        results = result_list[i]
        if results.shape[-1] > n_runs:
            results = results[:, :, -1]

        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        if mode == 'mean':
            values = np.nanmean(results, axis=-1)
        elif mode == 'max':
            values = np.nanmax(results, axis=-1)
        elif mode == 'min':
            values = np.nanmin(results, axis=-1)
        elif mode == 'median':
            values = np.nanmedian(results, axis=-1)

        for p, pos in enumerate(matrix_indices):
            heatmap[pos] = values[p]

        im = plt.imshow(heatmap, vmin=y_lim[0], vmax=y_lim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('# values', fontsize=fontsize)
        plt.ylabel('# attributes', fontsize=fontsize)
        plt.xticks(ticks=[0, 1, 2], labels=[4, 8, 16], fontsize=fontsize-1)
        plt.yticks(ticks=[0, 1, 2], labels=[3, 4, 5], fontsize=fontsize-1)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(y_lim)
        cbar.ax.tick_params(labelsize=fontsize-2)

        for k in range(3):
            for l in range(3):
                if not np.isnan(heatmap[l, k]):
                    ax = plt.gca()
                    _ = ax.text(l, k, np.round(heatmap[k, l], 2), ha="center", va="center", color="k",
                                fontsize=fontsize)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize+1, y=suptitle_position)

    plt.tight_layout()


def plot_heatmap_concept_x_context(result_list,
                                    mode,
                                    score,
                                    plot_dims=(2, 3),
                                    heatmap_size=(3, 3),
                                    figsize=(7, 7),
                                    ylims=(0.6, 1.0),
                                    titles=('D(3,4)', 'D(3,8)', 'D(3,16)', 'D(4,4)', 'D(4,8)', 'D(5,4)'),
                                    suptitle=None,
                                    suptitle_position=1.03,
                                    different_ylims=False,
                                    n_runs=5,
                                    matrix_indices=None,
                                    fontsize=18,
                                    inner_fontsize_fctr=0,
                                    one_dataset=False,
                                    attributes=4):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
    Allows for plotting multiple matrices according to plot_dims, and allows different modes:
    'max', 'min', mean', 'median', each across runs. """

    if score == 'NMI':
        score_idx = 0
    elif score == 'effectiveness':
        score_idx = 1
    elif score == 'consistency':
        score_idx = 2
    elif score == 'bosdis' or score == 'posdis':
        pass
    else:
        raise AssertionError("Score should be one of the following: 'NMI','effectiveness', 'consistency'.")

    plt.figure(figsize=figsize)

    # 6 datasets
    for i in range(np.prod(plot_dims)):
        # D(3,4), D(3,8), D(3,16)
        if titles[i] == 'D(3,4)' or titles[i] == 'D(3,8)' or titles[i] == 'D(3,16)':
            matrix_indices = sorted(list(itertools.product(range(3), repeat=2)), key=lambda x: x[1])
        # D(4,4), D(4,8)
        elif titles[i] == 'D(4,4)' or titles[i] == 'D(4,8)':
            matrix_indices = sorted(list(itertools.product(range(4), repeat=2)), key=lambda x: x[1])
        else:
            matrix_indices = sorted(list(itertools.product(range(5), repeat=2)), key=lambda x: x[1])
        if one_dataset:
            matrix_indices = sorted(list(itertools.product(range(attributes), repeat=2)), key=lambda x: x[1])

        if different_ylims:
            y_lim = ylims[i]
        else:
            y_lim = ylims

        heatmap = np.empty(heatmap_size)
        heatmap[:] = np.nan
        if score == 'bosdis' or score == 'posdis':
            results = result_list[i]
        else:
            results = result_list[score_idx][i]
            if results.shape[-1] > n_runs:
                results = results[:, :, -1]
            
        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        results_ls = [res.tolist() for res in results]

        if mode == 'mean':
            values = np.nanmean(results_ls, axis=0)
        elif mode == 'max':
            values = np.nanmax(results, axis=-1)
        elif mode == 'min':
            values = np.nanmin(results, axis=-1)
        elif mode == 'median':
            values = np.nanmedian(results, axis=-1)

        for p, pos in enumerate(matrix_indices):
            try:
                heatmap[pos] = values[p]
            except:
                IndexError

        im = plt.imshow(heatmap, vmin=y_lim[0], vmax=y_lim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('# Fixed Attributes', fontsize=fontsize)
        plt.ylabel('# Shared Attributes', fontsize=fontsize)
        plt.xticks(ticks=list(range(len(heatmap))), labels=list(range(1, len(heatmap)+1)), fontsize=fontsize-1)
        plt.yticks(ticks=list(range(len(heatmap))), labels=list(range(len(heatmap))), fontsize=fontsize-1)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(y_lim)
        cbar.ax.tick_params(labelsize=fontsize-2)

        for col in range(len(heatmap)):
            for row in range(len(heatmap[0])):
                if not np.isnan(heatmap[row, col]):
                    ax = plt.gca()
                    _ = ax.text(col, row, np.round(heatmap[row, col], 2), ha="center", va="center", color="k",
                                fontsize=fontsize+inner_fontsize_fctr)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize+1, y=suptitle_position)

    plt.tight_layout()


def plot_heatmap_concept_x_context_errors(result_list,
                                          plot_dims=(2, 3),
                                          heatmap_size=(3, 3),
                                          figsize=(7, 7),
                                          ylims=(0.6, 1.0),
                                          titles=('D(3,4)', 'D(3,8)', 'D(3,16)', 'D(4,4)', 'D(4,8)', 'D(5,4)'),
                                          datasets = ['(3,4)', '(3,8)', '(3,16)', '(4,4)', '(4,8)', '(5,4)'],
                                          suptitle=None,
                                          suptitle_position=1.03,
                                          different_ylims=False,
                                          n_runs=5,
                                          matrix_indices=None,
                                          fontsize=18,
                                          inner_fontsize_fctr=0,
                                          one_dataset=False,
                                          attributes=4):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
    Allows for plotting multiple matrices according to plot_dims.
    This function has been adapted to the use of displaying errors for each concept x context condition"""

    plt.figure(figsize=figsize)

    # 6 datasets
    for i in range(np.prod(plot_dims)):
        # D(3,4), D(3,8), D(3,16)
        if titles[i] == 'D(3,4)' or titles[i] == 'D(3,8)' or titles[i] == 'D(3,16)':
            matrix_indices = sorted(list(itertools.product(range(3), repeat=2)), key=lambda x: x[1])
        # D(4,4), D(4,8)
        elif titles[i] == 'D(4,4)' or titles[i] == 'D(4,8)':
            matrix_indices = sorted(list(itertools.product(range(4), repeat=2)), key=lambda x: x[1])
        else:
            matrix_indices = sorted(list(itertools.product(range(5), repeat=2)), key=lambda x: x[1])
        if one_dataset:
            matrix_indices = sorted(list(itertools.product(range(attributes), repeat=2)), key=lambda x: x[1])

        if different_ylims:
            y_lim = ylims[i]
        else:
            y_lim = ylims

        heatmap = np.empty(heatmap_size)
        heatmap[:] = np.nan

        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        for p, pos in enumerate(matrix_indices):
            try:
                if one_dataset:
                    heatmap[pos] = result_list[pos]
                else:
                    results = result_list[datasets[i]]
                    heatmap[pos] = results[pos]
            except:
                IndexError

        im = plt.imshow(heatmap, vmin=y_lim[0], vmax=y_lim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('# Fixed Attributes', fontsize=fontsize)
        plt.ylabel('# Shared Attributes', fontsize=fontsize)
        plt.xticks(ticks=list(range(len(heatmap))), labels=list(range(1, len(heatmap) + 1)), fontsize=fontsize - 1)
        plt.yticks(ticks=list(range(len(heatmap))), labels=list(range(len(heatmap))), fontsize=fontsize - 1)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(y_lim)
        cbar.ax.tick_params(labelsize=fontsize - 2)

        for col in range(len(heatmap)):
            for row in range(len(heatmap[0])):
                if not np.isnan(heatmap[row, col]):
                    ax = plt.gca()
                    _ = ax.text(col, row, np.round(heatmap[row, col], 2), ha="center", va="center", color="k",
                                fontsize=fontsize + inner_fontsize_fctr)

        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize + 1, y=suptitle_position)

    plt.tight_layout()



def plot_heatmap_different_vs(result_list,
                              mode,
                              plot_dims=(2, 2),
                              figsize=(7, 9),
                              ylims=(0.6, 1.0),
                              titles=('train', 'validation', 'zero shot objects', 'zero shot abstractions'),
                              suptitle=None,
                              suptitle_position=1.03,
                              n_runs=5,
                              matrix_indices=((0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)),
                              different_ylims=False,
                              fontsize=18,
                              ):
    """ Plot heatmaps in matrix arrangement for single values (e.g. final accuracies).
        Allows for plotting multiple matrices according to plot_dims, and allows different modes:
        'max', 'min', mean', 'median', each across runs.
    """

    plt.figure(figsize=figsize)

    for i in range(np.prod(plot_dims)):

        if different_ylims:
            ylim = ylims[i]
        else:
            ylim = ylims

        heatmap = np.empty((4, 2))
        heatmap[:] = np.nan
        results = result_list[i]
        if results.shape[-1] > n_runs:
            results = results[:, :, -1]

        plt.subplot(plot_dims[0], plot_dims[1], i + 1)

        if mode == 'mean':
            values = np.nanmean(results, axis=-1)
        elif mode == 'max':
            values = np.nanmax(results, axis=-1)
        elif mode == 'min':
            values = np.nanmin(results, axis=-1)
        elif mode == 'median':
            values = np.nanmedian(results, axis=-1)

        for p, pos in enumerate(matrix_indices):
            try:
                heatmap[pos] = values[p]
            except:
                continue

        im = plt.imshow(heatmap, vmin=ylim[0], vmax=ylim[1])
        plt.title(titles[i], fontsize=fontsize)
        plt.xlabel('balanced', fontsize=fontsize)
        plt.ylabel('vocab size factor', fontsize=fontsize)
        plt.xticks(ticks=[0, 1], labels=['True', 'False'], fontsize=fontsize-1)
        plt.yticks(ticks=[0, 1, 2, 3], labels=[1, 2, 3, 4], fontsize=fontsize-1)
        cbar = plt.colorbar(im, fraction=0.05, pad=0.04)
        cbar.ax.get_yaxis().set_ticks(ylim)
        cbar.ax.tick_params(labelsize=fontsize-2)

        for k in range(4):
            for l in range(2):
                if not np.isnan(heatmap[k, l]):
                    ax = plt.gca()
                    _ = ax.text(l, k, np.round(heatmap[k, l], 2), ha="center", va="center", color="k",
                                fontsize=fontsize)
        if suptitle:
            plt.suptitle(suptitle, fontsize=fontsize+1, x=0.51, y=suptitle_position)
    plt.tight_layout()


def plot_frequency_x_message_length(paths, setting, n_runs, n_values, datasets, n_epochs=300, color=['b', 'r'],
                                    optimal_color='g', mean_runs=True, smoothing=False, std=False, frequency='message',
                                    plot_frequency=False, int='train', labels=None, fontsize=16, ylim=None, xlim=None,
                                    yticks=None):
    """ This function creates a plot showing the message length as a function of the frequency rank.

    :param paths: list
    :param setting: list
    :param n_runs: int
    :param n_values: Iterable
    :param datasets: Iterable
    :param n_epochs: int
    :param color: Iterable with pyplot color codes with len(color) = (len(setting))
    :param optimal_color: str, Color for optimal coding, if None, there is no optimal coding
    :param mean_runs: If True calculate mean over runs, if false plot all runs
    :param smoothing: bool
    :param std: if mean_runs == True, whether to add an std around the mean
    :param frequency: str, one of 'message','input_concept','input_concept_x_context'
    :param plot_frequency: bool
    """

    max_rank_list = []
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 4 * len(datasets)), sharex=True, sharey=True)

    if len(datasets) == 1:
        axes = [axes]

    setting = [setting] if type(setting) == str else setting

    for i, (path, ax) in enumerate(zip(paths, axes)):

        if plot_frequency:
            ax2 = ax.twinx()

        for si, s in enumerate(setting):

            ordered_lengths_total = []
            ordered_frequencies_total = []

            for run in range(
                    n_runs):  # retrieve inputs instead of messages, but still need messages to calculate average length
                interaction = load_interaction(path, s, run, n_epochs, int=int)
                messages = retrieve_messages_freq_rank(interaction, is_gumbel=False, trim_eos=True, max_mess_len=21)
                if frequency == 'message':
                    unique_values, frequencies = torch.unique(messages, dim=0, return_counts=True)
                    message_length = MessageLengthHierarchical.compute_message_length(unique_values)
                elif frequency == 'input_concept' or frequency == 'input_concept_x_context':
                    concepts, context_conds = retrieve_concepts_context(interaction, n_values[0])
                    if frequency == 'input_concept':
                        inputs = torch.tensor([np.where(f, o[0], -1.) for o, f in concepts])
                    else:  # append context to input
                        inputs = torch.tensor(
                            [np.append(np.where(f, o[0], -1.), c) for (o, f), c in zip(concepts, context_conds)])
                    unique_values, frequencies = torch.unique(inputs, dim=0, return_counts=True)

                    message_length_dict = {tuple(uv.tolist()): [] for uv in unique_values}
                    for idx, input_val in enumerate(inputs):
                        message_length_dict[tuple(input_val.tolist())].append(
                            MessageLengthHierarchical.compute_message_length(messages[idx].unsqueeze(0)))
                    message_length = torch.tensor(
                        [torch.mean(torch.tensor(message_length_dict[tuple(uv.tolist())], dtype=torch.float32)).item()
                         for uv in unique_values])
                else:
                    return -1

                sorted_frequencies, sorted_indices = torch.sort(frequencies, descending=True)
                relative_sorted_frequencies = sorted_frequencies / sum(sorted_frequencies)
                ordered_lengths = message_length[sorted_indices]

                ordered_lengths_total.append(ordered_lengths)
                ordered_frequencies_total.append(relative_sorted_frequencies)

            max_rank = max([len(ordered_lengths_total[i]) for i in range(n_runs)])
            max_rank_list.append(max_rank)

            if mean_runs:
                ordered_lengths_mean = []
                ordered_frequencies_mean = []

                # calculate mean for each frequency rank
                for rank in range(max_rank):
                    c = [ordered_lengths_total[run][rank] for run in range(n_runs) if
                         len(ordered_lengths_total[run]) > rank]
                    ordered_lengths_mean.append(sum(c) / len(c))

                    if plot_frequency:
                        freq = [ordered_frequencies_total[run][rank] for run in range(n_runs) if
                                len(ordered_frequencies_total[run]) > rank]
                        ordered_frequencies_mean.append(sum(freq) / len(freq))

                if labels:
                    ax.plot(list(range(max_rank)), ordered_lengths_mean, color[si], label=labels[si])
                    if plot_frequency:
                        ax2.plot(list(range(max_rank)), ordered_frequencies_mean, color[si], label=labels[si], ls='-.')
                else:
                    ax.plot(list(range(max_rank)), ordered_lengths_mean, color[si], label=s)
                    if plot_frequency:
                        ax2.plot(list(range(max_rank)), ordered_frequencies_mean, color[si], label=s, ls='-.')

                if std:
                    # calculate std:
                    ordered_lengths_std = []
                    for rank in range(max_rank):
                        c = [ordered_lengths_total[run][rank] for run in range(n_runs) if
                             len(ordered_lengths_total[run]) > rank]
                        ordered_lengths_std.append(np.std(c))
                    ax.fill_between(list(range(max_rank)),
                                    np.array(ordered_lengths_mean) - np.array(ordered_lengths_std),
                                    np.array(ordered_lengths_mean) + np.array(ordered_lengths_std),
                                    color=color[si], alpha=0.3)
            else:
                for olt in ordered_lengths_total:
                    ax.plot(list(range(len(olt))), olt, color[si])
                if plot_frequency:
                    for oft in ordered_frequencies_total:
                        ax2.plot(list(range(len(olt))), oft, color[si], ls='-.')

        # optimal coding
        # if optimal_color != None:
        #    message_length_optimal = optimal_coding(vocab_size = (n_values[i] + 1)*3, nr_messages=max(max_rank_list))
        #    ax.plot(list(range(len(message_length_optimal))),message_length_optimal,optimal_color,ls='--',label=f"optimal coding {n_values[i]} values")

        ax.set_title(f"Dataset: {datasets[i]}", fontsize=fontsize+2)
        ax.legend(title='Message Length', fontsize=fontsize, loc='lower right', title_fontsize=fontsize+1)
        ax.set_xlabel('frequency rank', fontsize=fontsize+1)
        ax.set_ylabel('message length', fontsize=fontsize+1)
        ax.tick_params(axis='x', which='both', labelbottom=True)
        ax.tick_params(axis='both', labelsize=fontsize)
        if plot_frequency:
            ax2.set_ylabel('frequency count')
            ax2.legend(title='Frequency count', fontsize=fontsize, loc='upper right')

    if ylim:
        plt.ylim(ylim)
    if yticks:
        plt.yticks(yticks)
    if xlim:
        plt.xlim(xlim)

    plt.tight_layout()


def plot_frequency(paths, setting, n_runs, n_values, datasets, n_epochs=300, color=['b', 'r'], mean_runs=True,
                   std=False, frequency='message', int='train', labels=None, fontsize=16, ylim=None, yticks=None,
                   xlim=None, natural_language=False, linewidth=1):
    """ This function creates a plot showing the relative as a function of the frequency rank.

    :param paths: list
    :param setting: list
    :param n_runs: int
    :param n_values: Iterable
    :param datasets: Iterable
    :param n_epochs: int
    :param color: Iterable with pyplot color codes with len(color) = (len(setting))
    :param mean_runs: If True calculate mean over runs, if false plot all runs
    :param std: if mean_runs == True, whether to add an std around the mean
    :param frequency: str, one of 'message','input_concept','input_concept_x_context'
    """

    max_rank_list = []
    fig, axes = plt.subplots(len(datasets), 1, figsize=(8, 4 * len(datasets)), sharex=True, sharey=True)

    if len(datasets) == 1:
        axes = [axes]

    setting = [setting] if type(setting) == str else setting

    for i, (path, ax) in enumerate(zip(paths, axes)):

        for si, s in enumerate(setting):

            ordered_frequencies_total = []

            for run in range(
                    n_runs):  # retrieve inputs instead of messages, but still need messages to calculate average length
                interaction = load_interaction(path, s, run, n_epochs, int=int)
                messages = retrieve_messages_freq_rank(interaction, is_gumbel=False, trim_eos=True, max_mess_len=21)

                if frequency == 'message':
                    unique_values, frequencies = torch.unique(messages, dim=0, return_counts=True)
                elif frequency == 'input_concept' or frequency == 'input_concept_x_context':
                    concepts, context_conds = retrieve_concepts_context(interaction, n_values[0])
                    if frequency == 'input_concept':
                        inputs = torch.tensor([np.where(f, o[0], -1.) for o, f in concepts])
                    else:  # append context to input
                        inputs = torch.tensor(
                            [np.append(np.where(f, o[0], -1.), c) for (o, f), c in zip(concepts, context_conds)])
                    unique_values, frequencies = torch.unique(inputs, dim=0, return_counts=True)

                    message_length_dict = {tuple(uv.tolist()): [] for uv in unique_values}
                    for idx, input_val in enumerate(inputs):
                        message_length_dict[tuple(input_val.tolist())].append(
                            MessageLengthHierarchical.compute_message_length(messages[idx].unsqueeze(0)))
                else:
                    return -1

                sorted_frequencies, sorted_indices = torch.sort(frequencies, descending=True)
                relative_sorted_frequencies = sorted_frequencies / sum(sorted_frequencies)
                ordered_frequencies_total.append(relative_sorted_frequencies)

            max_rank = max([len(ordered_frequencies_total[i]) for i in range(n_runs)])
            max_rank_list.append(max_rank)

            if mean_runs:
                ordered_frequencies_mean = []

                # calculate mean for each frequency rank
                for rank in range(max_rank):
                    freq = [ordered_frequencies_total[run][rank] for run in range(n_runs) if
                            len(ordered_frequencies_total[run]) > rank]
                    ordered_frequencies_mean.append(sum(freq) / len(freq))

                if labels:
                    ax.plot(list(range(max_rank)), ordered_frequencies_mean, color[si], label=labels[si],
                            linewidth=linewidth)
                else:
                    ax.plot(list(range(max_rank)), ordered_frequencies_mean, color[si], label=s, linewidth=linewidth)

                if std:
                    # calculate std:
                    ordered_f_std = []
                    for rank in range(max_rank):
                        c = [ordered_frequencies_total[run][rank] for run in range(n_runs) if
                             len(ordered_frequencies_total[run]) > rank]
                        ordered_f_std.append(np.std(c))
                    ax.fill_between(list(range(max_rank)),
                                    np.array(ordered_frequencies_mean) - np.array(ordered_f_std),
                                    np.array(ordered_frequencies_mean) + np.array(ordered_f_std),
                                    color=color[si], alpha=0.3)
            else:
                for oft in ordered_frequencies_total:
                    ax.plot(list(range(len(oft))), oft, color[si], ls='-.', linewidth=linewidth)

        if natural_language:
            for l, language in enumerate(natural_language):
                language['relative frequency'] = language['frequency'] / sum(language['frequency'])
                ax.plot(language.index, language['relative frequency'], label=labels[si+l+1], color=color[si+l+1],
                        ls='dotted', linewidth=linewidth)

        ax.set_title(f"Dataset: {datasets[i]}", fontsize=fontsize+2)
        ax.set_xlabel('frequency rank', fontsize=fontsize+1)
        ax.tick_params(axis='x', which='both', labelbottom=True)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.set_ylabel('relative frequency', fontsize=fontsize+1)
        ax.legend(fontsize=fontsize, loc='lower right')

    if ylim:
        plt.ylim(ylim)
    if yticks:
        plt.yticks(yticks)
    if xlim:
        plt.xlim(xlim)

    plt.tight_layout()
