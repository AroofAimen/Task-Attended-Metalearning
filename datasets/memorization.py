import learn2learn as l2l
from learn2learn.data.transforms import TaskTransform
from learn2learn.data import DataDescription

import random
import functools

class XRemapLabels(TaskTransform):
    def __init__(self, dataset, shuffle=True):
        super(XRemapLabels, self).__init__(dataset)
        self.dataset = dataset
        self.shuffle = shuffle

    def remap(self, data, mapping):
        data = [d for d in data]
        data[1] = mapping(data[1])
        return data

    def __call__(self, task_description):
        if task_description is None:
            task_description = self.new_task()
        labels = list(set(self.dataset.indices_to_labels[dd.index] for dd in task_description))
        
        if self.shuffle:
            random.shuffle(labels)

        def mapping(x):
            # print("Mapping ",x," to ", (x%len(labels)))
            return int(x%len(labels))

        for dd in task_description:
            remap = functools.partial(self.remap, mapping=mapping)
            dd.transforms.append(remap)
        return task_description

class XNWays(TaskTransform):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/data/transforms.py)

    **Description**

    Keeps samples from N random labels present in the task description.

    **Arguments**

    * **dataset** (Dataset) - The dataset from which to load the sample.
    * **n** (int, *optional*, default=2) - Number of labels to sample from the task
        description's labels.

    """

    def __init__(self, dataset, n=2,filter_labels=None):
        super(XNWays, self).__init__(dataset)
        self.n = n
        self.indices_to_labels = dict(dataset.indices_to_labels)
        self.buckets = {}
        
        if filter_labels is None:
            filter_labels = self.dataset.labels
        self.filter_labels = filter_labels
        # print('self.filter_labels',self.filter_labels)
        for i in range(self.n):
            self.buckets[i] = []

        for class_id in self.filter_labels:
            self.buckets[class_id%n].append(class_id)
        # print('buckets', self.buckets)
        

    def new_task(self): 
        labels = self.filter_labels
        task_description = []
        labels_to_indices = dict(self.dataset.labels_to_indices)
        
        classes = [random.sample(self.buckets[i],k=1)[0] for i in range(self.n)]
        # print('classes', classes)
        # classes = random.sample(labels, k=self.n)
        
        for cl in classes:
            # print(cl,'labels_to_indices[cl]', labels_to_indices[cl])
            for idx in labels_to_indices[cl]:
                task_description.append(DataDescription(idx))
        return task_description

    def __call__(self, task_description):
        if task_description is None:
            # print("inside loop")
            return self.new_task()
        # print("outside loop")

        ## Exexutes when a new task is present##
        classes = []
        result = []
        set_classes = set()
        for dd in task_description:
            # print("dd",dd)
            set_classes.add(self.indices_to_labels[dd.index])
        classes = list(set_classes)
        classes = random.sample(classes, k=self.n)
        for dd in task_description:
            if self.indices_to_labels[dd.index] in classes:
                result.append(dd)
        return result


# cdef class CythonNWays(TaskTransform):

#     cdef public:
#         int n
#         dict indices_to_labels

#     def __init__(self, dataset, int n=2):
#         super(CythonNWays, self).__init__(dataset)
#         self.n = n
#         self.indices_to_labels = dict(dataset.indices_to_labels)

#     def __reduce__(self):
#         return CythonNWays, (self.dataset, self.n)

#     @cython.boundscheck(False)
#     @cython.wraparound(False)
#     cpdef new_task(self):  # Efficient initializer
#         cdef list labels = self.dataset.labels
#         cdef list task_description = []
#         cdef dict labels_to_indices = dict(self.dataset.labels_to_indices)
#         classes = random.sample(labels, k=self.n)
#         for cl in classes:
#             for idx in labels_to_indices[cl]:
#                 task_description.append(DataDescription(idx))
#         return task_description

#     def __call__(self, list task_description):
#         if task_description is None:
#             return self.new_task()
#         cdef list classes = []
#         cdef list result = []
#         cdef set set_classes = set()
#         cdef DataDescription dd
#         for dd in task_description:
#             set_classes.add(self.indices_to_labels[dd.index])
#         classes = <list>set_classes
#         classes = random.sample(classes, k=self.n)
#         for dd in task_description:
#             if self.indices_to_labels[dd.index] in classes:
#                 result.append(dd)
#         return result
