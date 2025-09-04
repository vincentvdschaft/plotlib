class Spacing:
    def __init__(self, horizontal, vertical):
        self.horizontal = float(horizontal)
        self.vertical = float(vertical)

    def __repr__(self):
        return f"Spacing(horizontal={self.horizontal}, vertical={self.vertical})"

    def to_tuple(self) -> tuple:
        """Return the spacing as a tuple."""
        return (self.horizontal, self.vertical)

    def to_dict(self) -> dict:
        """Return the spacing as a dictionary."""
        return {
            "horizontal": self.horizontal,
            "vertical": self.vertical,
        }
