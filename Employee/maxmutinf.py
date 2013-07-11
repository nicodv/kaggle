#!/usr/bin/env python
""" Maximum Mutual Information Criterion
"""
 
from math import log
 
class DataPoint(object):
    """ A single data point
 
    A data point associates a vector of values with a category
    """
    def __init__(self, vector, category):
        self.vector = vector
        self.category = category
 
    def __repr__(self):
        return "" % (self.vector, self.category)
 
    def __getitem__(self, i):
        """ The i-th value of the vector
        """
        return self.vector[i]
 
    def dimension(self):
        """ The dimensionality of the vector
        """
        return len(self.vector)
 
class DataSet(set):
    """ A set of data points
    """
    def __init__(self, data):
        super(DataSet, self).__init__()
        # The dimensionality of this data set. All data points must
        # have the same dimensionality.
        self.dimension = None
        # Set of all class values.
        self.categories = set()
        # Individual dimension vocabularies.
        self.vocabulary = []
        # Extract data ranges from the data.
        for d in data:
            d = DataPoint(*d)
            self.add(d)
            # Handle data point vector.
            if self.dimension == None:
                self.dimension = d.dimension()
                # If this is the first point, create the list of
                # dimension vocabularies.
                for i in xrange(self.dimension):
                    self.vocabulary.append(Set())
            elif not self.dimension == len(d.vector):
                # Ensure consistent dimensionality.
                raise ValueError, "Incorrect dimension in %s" % d
            # Handle data point category.
            self.categories.add(d.category)
            for i in xrange(d.dimension()):
                self.vocabulary[i].add(d[i])
 
    def __repr__(self):
        return "" % (self.__class__.__name__,len(self),len(self.categories),self.dimension)
 
    def p_joint(self, i, value, category):
        """ p(x[i] = value, category)
        """
        count = 0.0
        total = len(self)
        for d in self:
            if d[i] == value and d.category == category:
                count += 1
        return count/total
 
    def p_category_given_dimension(self, i, value, category):
        """ p(category | x[i] = value)
        """
        total = 0.0
        count = 0.0
        for d in self:
            if d[i] == value:
                total += 1
                if d.category == category:
                    count += 1
        return count/total
 
    def H_category_given_dimension(self, dim):
        """ H(category | x[i])
        """
        h = 0.0
        for c in self.categories:
            for value in self.vocabulary[dim]:
                h -= self.p_joint(dim, value, c) * \
                     log(self.p_category_given_dimension(dim, value, c), 2)
        return h
 
    def H_category_table(self):
        """ Table of the H(category | x[i]) in order of discrimativeness
        """
        h = {}
        for dim in xrange(self.dimension):
            h[dim] = self.H_category_given_dimension(dim)
        dims = h.keys()
        dims.sort(lambda a,b: cmp(b,a))
        lines = ["%d | %.5f" % (dim, h[dim]) for dim in dims]
        return "\n".join(lines)
 
    def p_joint_table(self, i):
        """ Table of p(x[i], category) for all x[i] and category
        """
        return self.table(self.p_joint, i)
 
    def conditional_probability_table(self, dim):
        """ Table of p(category | x[i]) for all x[i] and category
        """
        return self.table(self.p_category_given_dimension, dim)
 
    def table(self, func, i):
        """ Create a table given a component and an operation
        """
        # Sort the categories and vocabulary.
        vocabulary = list(self.vocabulary[i])
        vocabulary.sort()
        categories = list(self.categories)
        categories.sort()
        lines = []
        # Create header.
        header = ["%5s" % l for l in [""]+categories]
        header = ' | '.join(header)
        lines.append(header)
        # Create table.
        for value in vocabulary:
            line = ["%5s" % value] + ["%.3f" % func(i, value, c)
                                      for c in categories]
            line = ' | '.join(line)
            lines.append(line)
        return "\n".join(lines)
 
def sample_data_set():
    data = [([0,0], 'A'), # Class A
            ([1,0], 'A'),
            ([1,0], 'A'),
            ([0,0], 'A'),
            ([0,1], 'A'),
            ([0,1], 'B'), # Class B
            ([1,1], 'B'),
            ([1,1], 'B'),
            ([1,1], 'B'),
            ([0,0], 'B')
            ]
    return DataSet(data)
 
def underline(s):
    return "\n".join([s,'-'*len(s)])
 
if __name__ == "__main__":
    s = sample_data_set()
    for i in xrange(s.dimension):
        print "%s\n" % underline("Component %d" % i)
        print "Joint: p(x_%d, C)\n%s\n" % (i,s.p_joint_table(i))
        print "Conditional: p(C| x_%d )\n%s\n" % (i,s.conditional_probability_table(i))
    print "%s\n%s" % (underline("Conditional Entropy By Component"), s.H_category_table())
