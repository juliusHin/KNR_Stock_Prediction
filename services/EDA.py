from scipy import *


# The problem to optimize
from scipy._lib.six import xrange


def x2y2(x):
    return x[0] ** 2 + x[1] ** 2


class eda:
    def __init__(self, of):
        # Algorithm parameters
        self.iterations = 100
        self.sample_size = 100
        self.select_ratio = 0.5
        self.epsilon = 10e-6

        # class members
        self.objective_function = of
        self.dimensions = 2
        self.sample = []
        self.means = []
        self.stdevs = []
        self.debug = False

    def run(self):
        # uniform initialization
        self.sample = random.rand(self.sample_size, self.dimensions + 1)
        # cosmetic
        self.sample = self.sample * 200 - 100

        self.evaluate()

        # main loop
        i = 0
        while i < self.iterations:
            i += 1
            self.dispersion_reduction()
            self.estimate_parameters()
            self.draw_sample()
            self.evaluate()

        # sort the final sample
        self.sample_sort()
        # output the optimum
        print ("#[ x y f(x,y) ]")

        print (self.sample[0])


    def sample_sort(self):
        # sort rows on the last column
        self.sample = self.sample[argsort(self.sample[:, -1], 0)]

    def dispersion_reduction(self):
        self.sample_sort()

        # number of points to select
        nb = int(floor(self.sample_size * self.select_ratio))

        # selection
        self.sample = self.sample[:nb]

    def estimate_parameters(self):
        # points sub array (without values)
        mat = self.sample[:, :self.dimensions]

        # row means (axis 0 in scipy)
        self.means = mean(mat, 0)

        # row standard deviation
        self.stdevs = std(mat, 0)

    def draw_sample(self):
        # for each variable to optimize
        for i in xrange(self.dimensions):
            # if the dispersion is null
            if self.stdevs[i] == 0.0:
                # set it to a minimal value
                self.stdevs[i] = self.epsilon

        # empty sample
        self.sample = zeros((self.sample_size, self.dimensions + 1))

        # for each point
        for i in xrange(self.sample_size):
            # draw in random normal
            p = random.normal(self.means, self.stdevs)
            # put it into the sample
            self.sample[i][:self.dimensions] = p

    def evaluate(self):
        # for each point
        for i in xrange(self.sample_size):
            d = self.dimensions
            # call the objective function
            #   the third element is the result of the objective function call
            #   taking the first two elements as variables
            self.sample[i][-1] = self.objective_function(self.sample[i][:d])


if __name__ == "__main__":
    a = eda(x2y2)
    a.run()