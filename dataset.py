# code inspired by https://github.com/XeniaOhmer/hierarchical_reference_game/blob/master/dataset.py

import torch
import torch.nn.functional as F
import itertools
import random
from tqdm import tqdm

import numpy as np

SPLIT = (0.6, 0.2, 0.2)
SPLIT_ZERO_SHOT = (0.75, 0.25)


class DataSet(torch.utils.data.Dataset):
    """
    This class provides the torch.Dataloader-loadable dataset.
    """

    def __init__(self, properties_dim=[3, 3, 3], game_size=10, scaling_factor=10, device='cuda', testing=False,
                 sample_context=False, granularity="mixed", shared_context=False):
        """
        properties_dim: vector that defines how many attributes and features per attributes the dataset should contain,
        defaults to a 3x3x3 dataset
        game_size: integer that defines how many targets and distractors a game consists of
        """
        super().__init__()

        self.properties_dim = properties_dim
        self.game_size = game_size
        self.scaling_factor = scaling_factor
        self.device = device
        self.sample_context = sample_context
        self.granularity = granularity
        self.shared_context = shared_context

        # check if granularity has one of the allowed values
        if granularity not in ["mixed", "fine", "coarse"]:
            raise ValueError("Granularity must be one of: 'mixed', 'fine', or 'coarse'.")

        # check that sample context is not used together with fine or coarse granularities
        if sample_context and granularity in ["fine", "coarse"]:
            raise ValueError("Sample context can only be applied in the mixed granularity (standard) condition.")

        self.properties_dim = properties_dim
        self.all_objects = self._get_all_possible_objects(properties_dim)
        self.encoding_func = self._many_hot_encoding
        # get all concepts
        self.concepts = self.get_all_concepts()

        # generate dataset
        if not testing:
            self.dataset = self.get_datasets(split_ratio=SPLIT)

    def __len__(self):
        """Returns the total amount of samples in dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Returns the i-th sample (and label?) given an index (idx)."""
        return self.dataset[idx]

    def get_datasets(self, split_ratio, include_concept=False):
        """
        Creates the train, validation and test datasets based on the number of possible concepts.
        """
        if sum(split_ratio) != 1:
            raise ValueError

        train_ratio, val_ratio, test_ratio = split_ratio

        # Shuffle sender indices
        concept_indices = torch.randperm(len(self.concepts)).tolist()
        # Split is based on how many distinct concepts there are (regardless context conditions)
        ratio = int(len(self.concepts) * (train_ratio + val_ratio))

        train_and_val = []
        print("Creating train_ds and val_ds...")
        for concept_idx in tqdm(concept_indices[:ratio]):
            for _ in range(self.scaling_factor):
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                # for each concept, we can consider all possible context conditions -> "mixed" granularity
                # i.e. 1 for generic concepts, and up to len(properties_dim) for more specific concepts
                if not self.sample_context:
                    # standard: take all possible context conditions
                    if self.granularity == "mixed":
                        for context_condition in range(nr_possible_contexts):
                            train_and_val.append(
                                self.get_item(concept_idx, context_condition, self.encoding_func, include_concept))
                    # fine context condition has n_fixed-1 shared attributes between targets and distractors
                    # n.b. the non-shared attributes is *not* fixed
                    elif self.granularity == "fine":
                        train_and_val.append(
                            self.get_item(concept_idx, nr_possible_contexts - 1, self.encoding_func,
                                          include_concept))
                    # coarse context condition has no shared attributes between targets and distractors
                    elif self.granularity == "coarse":
                        train_and_val.append(
                            self.get_item(concept_idx, 0, self.encoding_func, include_concept))

                # or sample context condition from possible context conditions
                else:
                    context_condition = random.choice(range(nr_possible_contexts))
                    train_and_val.append(
                        self.get_item(concept_idx, context_condition, self.encoding_func, include_concept))

        # Calculating how many train
        train_samples = int(len(train_and_val) * (train_ratio / (train_ratio + val_ratio)))
        val_samples = len(train_and_val) - train_samples
        train, val = torch.utils.data.random_split(train_and_val, [train_samples, val_samples])
        # Save information about train dataset
        train.dimensions = self.properties_dim

        test = []
        print("\nCreating test_ds...")
        for concept_idx in tqdm(concept_indices[ratio:]):
            for _ in range(self.scaling_factor):
                nr_possible_contexts = sum(self.concepts[concept_idx][1])
                if not self.sample_context:
                    for context_condition in range(nr_possible_contexts):
                        test.append(
                            self.get_item(concept_idx, context_condition, self.encoding_func, include_concept))
                # or sample context condition from possible context conditions
                else:
                    context_condition = random.choice(range(nr_possible_contexts))
                    test.append(self.get_item(concept_idx, context_condition, self.encoding_func, include_concept))

        return train, val, test

    def get_item(self, concept_idx, context_condition, encoding_func, include_concept=False):
        """
        Receives concept-context pairs and an encoding function.
        Returns encoded (sender_input, labels, receiver_input).
            sender_input: (sender_input_objects, sender_labels)
            labels: indices of target objects in the receiver_input
            receiver_input: receiver_input_objects
        The sender_input_objects and the receiver_input_objects are different objects sampled from the same concept
        and context condition.
        """
        if self.shared_context:
            (sender_concept, sender_context,
             receiver_concept, receiver_context) = self.get_shared_context(concept_idx, context_condition)
        else:
            # use get_sample() to get sampled target and distractor objects
            # The concrete sampled objects can differ between sender and receiver.
            sender_concept, sender_context = self.get_sample(concept_idx, context_condition)
            receiver_concept, receiver_context = self.get_sample(concept_idx, context_condition)

        # subset such that only target objects are presented to sender and receiver
        sender_targets = sender_concept[0]
        receiver_targets = receiver_concept[0]
        sender_input = [obj for obj in sender_targets]
        receiver_input = [obj for obj in receiver_targets]
        # append context objects
        # get context of relevant context condition
        for distractor_objects, context_cond in sender_context:
            if context_cond == context_condition:
                # add distractor objects for the sender
                for obj in distractor_objects:
                    sender_input.append(obj)
        for distractor_objects, context_cond in receiver_context:
            if context_cond == context_condition:
                # add distractor objects for the receiver
                for obj in distractor_objects:
                    receiver_input.append(obj)
        # sender input does not need to be shuffled
        # shuffle receiver input and create (many-hot encoded) label
        random.shuffle(receiver_input)
        receiver_label = [idx for idx, obj in enumerate(receiver_input) if obj in receiver_targets]
        receiver_label = torch.Tensor(receiver_label).to(torch.int64).to(device=self.device)
        receiver_label = F.one_hot(receiver_label, num_classes=self.game_size * 2).sum(dim=0).float()
        # ENCODE and return as TENSOR
        sender_input = torch.stack([encoding_func(elem) for elem in sender_input])
        receiver_input = torch.stack([encoding_func(elem) for elem in receiver_input])
        # output needs to have the structure sender_input, labels, receiver_input
        return sender_input, receiver_label, receiver_input

    def get_sample(self, concept_idx, context_condition):
        """
        Returns a full sample consisting of a set of target objects (target concept)
        and a set of distractor objects (context) for a given concept condition.
        To be used separately for sender and receiver.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        # sample target objects for given game size (if possible, get unique choices)
        try:
            target_objects = random.sample(all_target_objects, self.game_size)
        except ValueError:
            target_objects = random.choices(all_target_objects, k=self.game_size)
        # get all possible distractors for a given concept (for all possible context conditions)
        context = self.get_distractors(concept_idx, context_condition)
        context_sampled = self.sample_distractors(context, context_condition)
        # return target concept, context (distractor objects + context_condition) for each context
        return [target_objects, fixed], context_sampled

    def get_shared_context(self, concept_idx, context_condition):
        """
        Returns a full sample consisting of a set of target objects (target concept)
        and a set of distractor objects (context) for a given concept condition.
        This can be used to create a shared context between sender and receiver.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        for interlocutor in ['sender', 'receiver']:
            # sample target objects for given game size (if possible, get unique choices)
            try:
                target_objects = random.sample(all_target_objects, self.game_size)
            except ValueError:
                target_objects = random.choices(all_target_objects, k=self.game_size)
            # get all possible distractors for a given concept (for all possible context conditions)
            if interlocutor == 'sender':
                context, shared_attr_indices = self.get_distractors_shared(concept_idx, context_condition)
            else:
                context, _ = self.get_distractors_shared(concept_idx, context_condition, shared_attr_indices)

            context_sampled = self.sample_distractors(context, context_condition)

            if interlocutor == 'sender':
                s_target_objects = target_objects
                s_context_sampled = context_sampled
            else:
                r_target_objects = target_objects
                r_context_sampled = context_sampled

        # return target concept, context (distractor objects + context_condition) for each context
        return [s_target_objects, fixed], s_context_sampled, [r_target_objects, fixed], r_context_sampled

    def get_distractors(self, concept_idx, context_condition):
        """
        Computes distractors.
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        context = []

        # save fixed attribute indices in a list for later comparisons
        fixed_attr_indices = []
        for index, value in enumerate(fixed):
            if value == 1:
                fixed_attr_indices.append(index)

        # consider all objects as possible distractors
        poss_dist = self.all_objects

        for obj in poss_dist:
            # find out how many attributes are shared between the possible distractor object and the target concept
            # (by only comparing fixed attributes because only these are relevant for defining the context)
            shared = sum(1 for idx in fixed_attr_indices if obj[idx] == all_target_objects[0][idx])
            if shared == context_condition:
                context.append(obj)

        return context

    def get_distractors_shared(self, concept_idx, context_condition, shared_attr_indices=None):
        """
        Computes distractors for a shared context between sender and receiver. It also implements a change compared
        to get_distractors(): Here, objects in the context do not only share a certain number of attributes, but also
        the position of the attributes that are shared are fixed (and shared between sender and receiver).
        """
        all_target_objects, fixed = self.concepts[concept_idx]
        context = []

        # save fixed attribute indices in a list for later comparisons
        fixed_attr_indices = []
        for index, value in enumerate(fixed):
            if value == 1:
                fixed_attr_indices.append(index)

        # if not given, compute (this is the case for the sender)
        if not shared_attr_indices:
            # generate a shared vector, i.e. a vector that indicates which of the attributes should be shared with the
            # concept attributes according to the context condition
            # e.g. if fixed==(1,1,1) and context_condition==2, then shared vector can be one of: (1,1,0), (1,0,1), (0,1,1)
            # which attributes should be shared:
            shared_attr_indices = random.sample(fixed_attr_indices, context_condition)

        # consider all objects as possible distractors
        poss_dist = self.all_objects

        for obj in poss_dist:
            # find out how many attributes are shared between the possible distractor object and the target concept
            # (by only comparing fixed attributes because only these are relevant for defining the context)
            shared = sum(1 for idx in fixed_attr_indices if obj[idx] == all_target_objects[0][idx])
            # this still works for context condition 0:
            if len(shared_attr_indices) == 0:
                if shared == context_condition:
                    context.append(obj)
            else:
                if shared == context_condition:  # this filters out already some of the objects to keep the loop as small as possible
                    if np.all(
                            np.array(obj)[shared_attr_indices] == np.array(all_target_objects[0])[shared_attr_indices]):
                        context.append(obj)

        return context, shared_attr_indices

    def sample_distractors(self, context, context_condition):
        """
        Function for sampling the distractors from a specified context condition.
        """
        # sample distractor objects for given game size and the specified context condition
        # distractors = [dist_obj for dist_objs in context for dist_obj in dist_objs]
        context_new = []
        try:
            context_new.append([random.sample(context, self.game_size), context_condition])
        except ValueError:
            context_new.append([random.choices(context, k=self.game_size), context_condition])
        return context_new

    def get_all_concepts(self):
        """
        Returns all possible concepts for a given dataset size.
        Concepts consist of (objects, fixed) tuples
            objects: a list with all object-tuples that satisfy the concept
            fixed: a tuple that denotes how many and which attributes are fixed
        """
        fixed_vectors = self.get_fixed_vectors(self.properties_dim)
        # create all possible concepts
        all_fixed_object_pairs = list(itertools.product(self.all_objects, fixed_vectors))

        concepts = list()
        # go through all concepts (i.e. fixed, objects pairs)
        for concept in all_fixed_object_pairs:
            # treat each fixed_object pair as a target concept once
            # e.g. target concept (_, _, 0) (i.e. fixed = (0,0,1) and objects e.g. (0,0,0), (1,0,0))
            fixed = concept[1]
            # go through all objects and check whether they satisfy the target concept (in this example have 0 as 3rd attribute)
            target_objects = list()
            for object in self.all_objects:
                if self.satisfies(object, concept):
                    if object not in target_objects:
                        target_objects.append(object)
            # concepts are tuples of fixed attributes and all target objects that satisfy the concept
            if (target_objects, fixed) not in concepts:
                concepts.append((target_objects, fixed))
        return concepts

    def get_shared_vectors(self, fixed):
        """
        Returns fixed vectors for all possible context conditions based on a concept (i.e. the fixed vector).
        These are called "shared_vectors" because the number and position of attributes which are shared with the
        target concept define the context condition. The more fixed attributes are shared, the finer the context.
        """
        shared_vectors = []
        for i, attribute in enumerate(fixed):
            shared = list(itertools.repeat(0, len(fixed)))
            if attribute == 1:
                shared[i] = 1
                shared_vectors.append(shared)
        return shared_vectors


    @staticmethod
    def satisfies(object, concept):
        """
        Checks whether an object satisfies a target concept, returns a boolean value.
        Concept consists of an object vector and a fixed vector tuple.
        """
        satisfied = False
        same_counter = 0
        concept_object, fixed = concept
        # an object satisfies a concept if fixed attributes are the same
        # go through attributes an check whether they are fixed
        for i, attr in enumerate(fixed):
            # if an attribute is fixed
            if attr == 1:
                # compare object with concept object
                if object[i] == concept_object[i]:
                    same_counter = same_counter + 1
        # the number of shared attributes should match the number of fixed attributes
        if same_counter == sum(fixed):
            satisfied = True
        return satisfied

    @staticmethod
    def get_fixed_vectors(properties_dim):
        """
        Returns all possible fixed vectors for a given dataset size.
        Fixed vectors are vectors of length len(properties_dim), where 1 denotes that an attribute is fixed, 0 that it isn't.
        The more attributes are fixed, the more specific the concept -- the less attributes fixed, the more generic the concept.
        """
        # what I want to get: [(1,0,0), (0,1,0), (0,0,1)] for most generic
        # concrete: [(1,1,0), (0,1,1), (1,0,1)]
        # most concrete: [(1,1,1)]
        # for variable dataset sizes

        # range(0,2) because I want [0,1] values for whether an attribute is fixed or not
        list_of_dim = [range(0, 2) for dim in properties_dim]
        fixed_vectors = list(itertools.product(*list_of_dim))
        # remove first element (0,..,0) as one attribute always has to be fixed
        fixed_vectors.pop(0)
        return fixed_vectors

    @staticmethod
    def get_all_objects_for_a_concept(properties_dim, features, fixed):
        """
        Returns all possible objects for a concept at a given level of abstraction
        features: Defines the features which are fixed
        fixed: Defines how many and which attributes are fixed
        """
        # retrieve all possible objects
        list_of_dim = [range(0, dim) for dim in properties_dim]
        all_objects = list(itertools.product(*list_of_dim))

        # get concept objects
        concept_objects = list()

        # account for the case where 0 attributes should be shared in context_condition 0
        if not 1 in fixed:
            return all_objects

        # determine the indices of attributes that should be fixed
        fixed_indices = list(itertools.compress(range(0, len(fixed)), fixed))
        # find possible concepts for each index
        possible_concepts = dict()
        for index in fixed_indices:
            possible_concepts[index] = ([object for object in all_objects if object[index] == features[index]])

        # keep only those that also match with the other fixed features, i.e. that are possible concepts for all fixed indices
        all = list(possible_concepts.values())
        concept_objects = list(set(all[0]).intersection(*all[1:]))

        return concept_objects

    @staticmethod
    def _get_all_possible_objects(properties_dim):
        """
        Returns all possible combinations of attribute-feature values as a dataframe.
        """
        list_of_dim = [range(0, dim) for dim in properties_dim]
        # Each object is a row
        all_objects = list(itertools.product(*list_of_dim))
        return all_objects  # pd.DataFrame(all_objects)

    def _many_hot_encoding(self, input_list):
        """
        Outputs a binary one dim vector
        """
        output = torch.zeros([sum(self.properties_dim)]).to(device=self.device)
        start = 0

        for elem, dim in zip(input_list, self.properties_dim):
            output[start + elem] = 1
            start += dim

        return output

    def _sample_image_from_concept(self, concept):
        all_objects = self.reverse_one_hot()
        indices = np.where(np.all(all_objects == np.array(concept), axis=1))[0].tolist()
        random_index = random.choice(indices) if indices else None
        sampled_img = self.images[random_index] if random_index is not None else None
        return torch.tensor(sampled_img, dtype=torch.float32, device=self.device)

