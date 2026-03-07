from abc import ABC, abstractmethod
import pandas as pd

class Strategy(ABC):
    def __init__(self, name, params=None):
        self.name = name
        self.params = params if params else {}

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INPUT: DataFrame con precios (Open, High, Low, Close).
        OUTPUT: DataFrame original + columna 'Signal'.
                Signal = 1 (Comprar/Mantener)
                Signal = 0 (Vender/Efectivo)
                Signal = -1 (Corto - opcional)
        """
        pass