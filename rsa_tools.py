import itertools
import torch


def get_utterances(vocab_size, max_length, interactions=None, limit_utterances=0):
    if interactions != [None]:
        utterances = get_unique_utterances(interactions)
    else:
        print(f'Generating utterances with vocab size {vocab_size} and max length {max_length}')
        utterances = generate_utterances(vocab_size, max_length)

    print(f'Shape of utterances: {utterances.shape}')

    # Convert to one-hot encoding
    utterances = torch.nn.functional.one_hot(utterances, num_classes=vocab_size).float()

    # Randomly sample utterances up to limit_utterances if given
    if limit_utterances > 0:
        num_random_samples = min(limit_utterances, utterances.shape[0])
        random_indices = torch.randperm(utterances.shape[0])[:num_random_samples]
        utterances = utterances[random_indices]

        print(f'Randomly sampled {limit_utterances} utterances. New shape: {utterances.shape}')

    return utterances


def generate_utterances(vocab_size, max_length):
    """
    Generate all possible utterances using vocab_size and max_length
    """
    all_possible_utterances = []
    # Max length plus one for EOS symbol
    total_length = max_length + 1
    # Loop through each possible utterance length (1 to max_length)
    for length in range(1, max_length + 1):
        # Generate all combinations for current length
        for combination in itertools.product(range(1, vocab_size), repeat=length):
            # Append EOS
            utterance = list(combination) + [0]

            # Apply padding to the right for shorter utterances
            padding = [0] * (total_length - len(utterance))
            utterance = utterance + padding

            all_possible_utterances.append(utterance)
    return torch.tensor(all_possible_utterances)


def get_unique_utterances(interactions):
    """
    Get unique utterances from interaction data
    """
    all_messages = []

    for interaction in interactions:
        messages = interaction.message.argmax(dim=-1)
        messages = [msg.tolist() for msg in messages]
        all_messages.extend(messages)

    unique_messages = [list(x) for x in set(tuple(x) for x in all_messages)]

    return torch.tensor(unique_messages)
