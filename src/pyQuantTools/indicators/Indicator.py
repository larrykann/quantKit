from abc import ABC, abstractmethod

class Indicator(ABC):
    def __init__(self, data: dict):
        """
        Base class for indicators.

        Parameters:
        - data (dict): A dictionary containing the input data required for the indicator calculation.
          Example: {'high': np.ndarray, 'low': np.ndarray, 'close': np.ndarray}
        """
        self.data = data
        self.result = None
    
    @abstractmethod
    def calculate(self):
        """
        Placeholder method to be implemented by subclasses for specific indicator calculations.
        """
        raise NotImplementedError("Subclasses should implement this method.")

