class Point:
    # n_dimensions = 0  # number of dimensions
    # vec = []  # vector of dimension components
    # ID = ""  # point identifier
    max_ID = -1

    def __init__(self, components, ID=""):
        self.vec = components
        self.n_dimensions = len(components)
        if ID == "":
            self.ID = str(Point.max_ID + 1)
            Point.max_ID += 1
        else:
            self.ID = str(ID)
            if int(ID) > Point.max_ID:
                Point.max_ID = int(ID)
        return

    def __hash__(self):
        # return hash((self.ID, sum(self.vec)))
        return hash(self.ID)

    def __eq__(self, other):
        return self.ID == other.ID

    def __ne__(self, other):
        return not (self.ID == other.ID)

    def __str__(self):
        s = "< id:" + str(self.ID) + '; ['
        for d in range(self.n_dimensions - 1):
            s += str(self.vec[d]) + ","
        return s + str(self.vec[self.n_dimensions - 1]) + "]>"

    def at(self, d):
        return self.vec[d]
