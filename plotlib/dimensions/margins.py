class Margins:
    def __init__(self, left, right, top, bottom):
        for val in (left, right, top, bottom):
            Margins._check_input(val)

        self.left = float(left)
        self.right = float(right)
        self.top = float(top)
        self.bottom = float(bottom)

    @staticmethod
    def _check_input(value):
        if not isinstance(value, (int, float)):
            raise TypeError("Margin values must be numeric")

        if value < 0:
            raise ValueError("Margin values must be non-negative")

    @property
    def width(self):
        return self.left + self.right

    @property
    def height(self):
        return self.top + self.bottom

    def copy(self):
        return Margins(self.left, self.right, self.top, self.bottom)
