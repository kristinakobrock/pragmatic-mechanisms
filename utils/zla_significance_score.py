import torch
from utils.concept_reps import *

def mean_message_length_from_interaction(interaction, remove_after_eos=True):
    """ Calculates the average message length, with only accounting for each individual message one time.

    :param interaction: interaction (EGG class)
    """
    # messages = interaction.message.argmax(dim=-1)
    messages = retrieve_messages_freq_rank(interaction, is_gumbel=False, trim_eos=True)
    unique_messages = torch.unique(messages, dim=0)
    message_length = MessageLengthHierarchical.compute_message_length(unique_messages)
    av_message_length = torch.mean(message_length.float())
    return av_message_length


def mean_weighted_message_length_from_interaction(interaction):
    """ average length of the messages weighted by their generation frequency (how often does the agent generate this message)

    :param interaction: interaction (EGG class)
    """
    messages = retrieve_messages_freq_rank(interaction, is_gumbel=False, trim_eos=True)
    message_length = MessageLengthHierarchical.compute_message_length(messages)
    av_message_length = torch.mean(message_length.float())
    return av_message_length


def mean_weighted_message_length(length,
                                 frequency):  # TODO analyse context x concept counts and if it matches this calculation but should
    """ calculates the average length of the messages wieghted by their generation frequency from length and frequency.

    :param length: torch.Tensor with shape = (n,) contains the length of each message
    :param frequency: torch.Tensor with shape = (n,) contains the frequency of each message
    """
    return torch.sum(((length * frequency) / torch.sum(frequency)).float())


def mean_message_length_context_dep(interaction, values):
    """ Calculates the average message length, with all messages used in a certain context condition

    :param interaction: interaction (EGG class)
    :param values: int
    """
    messages = retrieve_messages_freq_rank(interaction, is_gumbel=False, trim_eos=True)
    message_length = MessageLengthHierarchical.compute_message_length(messages)
    _, context_condition = retrieve_concepts_context(interaction, values)

    context_dep_lengths = []
    for c in range(max(context_condition) + 1):
        single_context_length = message_length[torch.tensor(context_condition) == c].float()
        context_dep_lengths.append(single_context_length)
    message_length_step = [round(torch.mean(context_dep_lengths[i]).item(), 3) for i in
                           range(max(context_condition) + 1)]
    return message_length_step


def mean_message_length_context_x_concept(interaction, values):
    """ Calculates the average message length, with all messages used in a certain context x concept condition

    :param interaction: interaction (EGG class)
    :param values: int
    """
    """ Calculates the average message length, with all messages used in a certain context x concept condition

    :param interaction: interaction (EGG class)
    :param values: int
    """
    messages = retrieve_messages_freq_rank(interaction, is_gumbel=False, trim_eos=True)
    message_length = MessageLengthHierarchical.compute_message_length(messages)
    concepts, context_condition = retrieve_concepts_context(interaction, values)
    fixed_num = torch.tensor([np.sum(f) for c, f in concepts]).int()
    context = torch.tensor(context_condition)

    list_context_x_fixed = []
    for c in range(max(context_condition) + 1):
        list_fixed = []
        single_context_length = message_length[context == c].float()
        single_context_fixed = fixed_num[context == c]

        for f in range(fixed_num.max().int() + 1):
            single_context_fixed_length = single_context_length[single_context_fixed == f]

            list_fixed.append(round(torch.mean(single_context_fixed_length).item(), 3))
        list_context_x_fixed.append(list_fixed)

    return list_context_x_fixed


def count_message_lengths(interaction, max_length):
    """ return the occuring message lengths and their frequency """
    messages = retrieve_messages_freq_rank(interaction, is_gumbel=False, trim_eos=True)
    which, frequencies = torch.unique(MessageLengthHierarchical.compute_message_length(messages), return_counts=True)

    output = torch.zeros((max_length,), dtype=frequencies.dtype)
    for w, f in zip(which, frequencies):
        output[w] = f
    return (list(range(max_length)), output)


def ZLA_significance_score(interaction, values, num_permutations=1000, remove_after_eos=True):
    """ Calculates to which degree mean_weighted_message_length is lower than a random permutation of its frequency length mapping

    :param interaction: interaction (EGG class)
    :param values: int
    :param num_permutations: int
    :param remove_after_eos: bool (should be true, as two messages should be the same if the only difference is after eos)
    """

    # get message frequencies and message_length
    messages = retrieve_messages_freq_rank(interaction, is_gumbel=False, trim_eos=True)
    unique_messages, frequencies = torch.unique(messages, dim=0, return_counts=True)
    message_length = MessageLengthHierarchical.compute_message_length(unique_messages)

    L_type = torch.mean(message_length.float())  # mean message length
    original_L_token = mean_weighted_message_length_from_interaction(interaction)

    # random permutations of frequency-length mapping, then calulate their L_token
    permuted_L_tokens = []
    for _ in range(num_permutations):
        permuted_indices = torch.randperm(message_length.shape[0])
        permuted_lengths = message_length[permuted_indices]
        permuted_L_token = mean_weighted_message_length(permuted_lengths, frequencies)
        permuted_L_tokens.append(permuted_L_token)

    # calculate p_value
    bool_list = (torch.tensor(permuted_L_tokens) <= original_L_token)
    pZLA = bool_list.double().mean()

    # calculate weighted message length context x concept
    message_length_context_dep = mean_message_length_context_dep(interaction, values)
    messages_length_context_x_concept = mean_message_length_context_x_concept(interaction, values)
    message_length_frequency = count_message_lengths(interaction, max_length=len(messages[0]))

    score_dict = {'mean_message_length': L_type.tolist(),
                  'mean_weighted_message_length': original_L_token.tolist(),
                  'p_zla': pZLA.tolist(),
                  'min_bool_value': int(bool_list.min().tolist()),
                  'max_bool_value': int(bool_list.max().tolist()),
                  'message_length_context_dep': message_length_context_dep,
                  'message_length_context_x_concept': messages_length_context_x_concept,
                  'message_length_frequency': message_length_frequency}
    return score_dict