from .base import BaseClass
import json
import numpy as np
import os


class SingleMonitor(BaseClass):
    """
    Monitor that provides functionality to store a single value and (e.g. time) and keep track of statistics
    """
    def __init__(self, **kwargs):
        """
        Initialize the SingleMonitor object.
        """
        if "elements" not in kwargs:
            kwargs["elements"] = []
        super().__init__(**kwargs)
        
    def pop_results(self, n=1):
        for _ in range(n):
            self.elements.pop()
            
    def merge(self, other):
        """
        Merge the elements of another SingleMonitor object with the elements of this object.
        Args:
            other (SingleMonitor): The other SingleMonitor object.
        """
        self.elements.extend(other.elements)
        
    def add_result(self, element):
        """
        Add a result to the monitor.
        
        Args:
            element (float): The result to be added.
        """
        self.elements.append(element)
        
    def n_calls(self):
        """
        Returns the number of calls to the monitor.
        """
        return len(self.elements)
    
    def mean(self):
        """
        Returns the mean of the elements.
        """
        return float(np.mean(self.elements))
    
    def std(self):
        """
        Returns the standard deviation of the elements.
        """
        return float(np.std(self.elements))
    
    def total(self):
        """
        Returns the sum of the elements.
        """
        return float(np.sum(self.elements))
    
    def get_store_settings(self):
        """
        Return a dictionary containing the number of calls, mean, standard deviation, and total of the elements list.
        """
        return {
            "n_calls": self.n_calls(),
            "mean": self.mean(),
            "std": self.std(),
            "total": self.total()
        }
        
    def store(self, path):
        """
        Stores the object in a json file.
        
        Args:
            path (string): The file path at which the settings have to be saved.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        settings = self.get_store_settings()
        # store with json
        with open(path, "w") as f:
            json.dump(settings, f, indent=4, sort_keys=False)
            
class MultipleMonitor(SingleMonitor):
    """
    Monitor that allows for storing multiple values (e.g. time) and keep track of their statistics
    """
    def __init__(self, **kwargs):
        """
        Initialize the MultipleMonitor object.
        
        Args:
            kwargs (dict): The keyword arguments.
        """
        if "other_elements" not in kwargs:
            kwargs["other_elements"] = dict()
        super().__init__(**kwargs)
        
    def merge(self, other):
        """
        Merge the elements of another MultipleMonitor object with the elements of this object.
        Args:
            other(MultipleMonitor): The other MultipleMonitor object.
        """
        super().merge(other)
        for indicator, monitor in other.other_elements.items():
            if indicator not in self.other_elements:
                self.other_elements[indicator] = SingleMonitor()
            self.other_elements[indicator].merge(monitor)
    
    def add_time_type(self, indicator):
        """
        Add a new time type to the monitor.
        
        Args:
            indicator (string): The name of the time type.
        """
        self.other_elements[indicator] = SingleMonitor()
        
    def pop_results(self, n=1, indicator=None):
        """Pop results from the monitor.

        Args:
            n (int, optional): Number of elements to pop. Defaults to 1.
            indicator (string, optional): Name of the SingleMonitor from which to pop the elements. Defaults to None.
        """
        if indicator is None:
            super().pop_results(n)
        elif indicator in self.other_elements:
            self.other_elements[indicator].pop_results(n)
        
    def add_result(self, element, indicator=None):
        """
        Add a result to the monitor.
        
        Args:
            element (float): The result to be added.
            indicator (string, optional): The name of the time type.
        """
        if indicator is None:
            super().add_result(element)
        else:
            if indicator not in self.other_elements:
                self.add_time_type(indicator)
            self.other_elements[indicator].add_result(element)
            
    def get_store_settings(self):
        """
        Return a dictionary containing the parent class's store settings and the store settings of each SingleMonitor instance in the other_elements dictionary.
        """
        return {
            **super().get_store_settings(),
            **{indicator: monitor.get_store_settings() for indicator, monitor in self.other_elements.items()}
        }
    
    
class ModelMonitor(MultipleMonitor):
    """
    Keeps track of values associated with a specific runnable operator
    """
    def __init__(self, runnable_operator):
        """
        Initialize the ModelMonitor object.
        Args:
            runnable_operator (RunnableOperator): The runnable operator associated with the monitor.
        """
        super().__init__(runnable_operator=runnable_operator)
    
    def get_store_settings(self):
        """
        Gets the store settings of the parent class and the runnable operator.
        """
        return {
            **super().get_store_settings(),
            **self.runnable_operator.get_store_params()
        }
            
class ModelsMonitor(BaseClass):
    """
    Monitor for all runnable operators in the formula
    """
    def __init__(self, runnable_operators):
        """"
        Initialize the ModelsMonitor object.
        Args:
            runnable_operators (List[RunnableOperator]): A list of runnable operators.
        """
        self.monitors = {runnable_operator.id(): ModelMonitor(runnable_operator) for runnable_operator in runnable_operators}
        super().__init__(monitors=self.monitors)
        
    def merge(self, other):
        """
        Merge the elements of another ModelsMonitor object with the elements of this object.
        
        Args:
            other (ModelsMonitor): The other ModelsMonitor object.
        """
        for runnable_operator, monitor in other.monitors.items():
            if runnable_operator not in self.monitors:
                self.monitors[runnable_operator] = ModelMonitor(monitor.runnable_operator)
            self.monitors[runnable_operator].merge(monitor)
        
    def pop_results(self, n=1, runnable_operator=None, indicator=None):
        """Pop results from the monitor.

        Args:
            n (int, optional): Number of elements to pop. Defaults to 1.
            runnable_operator (RunnableOperator, optional): From which ModelMonitor to pop the results. Defaults to None.
            indicator (string, optional): Name of the type to pop. Defaults to None.
        """
        if runnable_operator is None:
            super().pop_results(n, indicator=indicator)
        else:
            self.monitors[runnable_operator.id()].pop_results(n, indicator=indicator)
    
    def add_result(self, element, runnable_operator, indicator=None):
        """
        Add a result to the monitor.
        Args:
            element (float): The result to be added.
            runnable_operator (RunnableOperator): The runnable operator associated with the result.
            indicator (string, optional): The name of the time type.
        """
        self.monitors[runnable_operator.id()].add_result(element, indicator)
        
    def get_store_settings(self):
        """
        Gets the store settings of each ModelMonitor instance in the monitors dictionary.
        """
        return [
            monitor.get_store_settings() for monitor in self.monitors.values()
        ]
        
    def store(self, path):
        """
        Stores the object in a json file.
        
        Args:
            path (string): The file path at which the settings have to be saved.
        """
        store_times = self.get_store_settings()
        # store with json
        with open(path, "w") as f:
            json.dump(store_times, f, indent=4, sort_keys=False)


class Monitor(MultipleMonitor):
    """
    Final monitor object that keeps track of values for runnable operators, but also for the whole formula
    """
    def __init__(self, runnable_operators):
        """
        Initialize the Monitor object.
        
        Args:
            runnable_operators(List[RunnableOperator]): A list of runnable operators.
        """
        super().__init__(models_monitor=ModelsMonitor(runnable_operators))
        
    def pop_results(self, n=1, runnable_operator=None, indicator=None):
        """Pop results from the monitor.

        Args:
            n (int, optional): Number of elements to pop. Defaults to 1.
            runnable_operator (RunnableOperator, optional): From which ModelMonitor to pop the results. Defaults to None.
            indicator (string, optional): Name of the type to pop. Defaults to None.
        """
        if runnable_operator is None:
            super().pop_results(n, indicator=indicator)
        else:
            self.models_monitor.pop_results(n, runnable_operator, indicator=indicator)
    
    def merge(self, other):
        """
        Merge the elements of another Monitor object with the elements of this object.
        Args:
            other (Monitor): The other Monitor object.
        """
        super().merge(other)
        self.models_monitor.merge(other.models_monitor)
        
    def add_result(self, element, runnable_operator=None, indicator=None):
        """
        Add a result to the monitor.
        Args:
            element (float): The result to be added.
            runnable_operator (RunnableOperator): The runnable operator associated with the result.
            indicator (string, optional): The name of the time type.
        """
        if runnable_operator is None:
            super().add_result(element, indicator=indicator)
        else:
            self.models_monitor.add_result(element, runnable_operator, indicator=indicator)
        
    def get_store_settings(self):
        """
        Gets the store settings of the parent class and the models monitor.
        """
        sum_vals = [monitor.total() for monitor in self.models_monitor.monitors.values()]
        if len(sum_vals) > 0:
            total_time_no_model_calls = self.total() - sum(sum_vals)
        else:
            total_time_no_model_calls = self.total()

        return {
            **super().get_store_settings(),
            "total_time_no_model_calls": total_time_no_model_calls,
            "models_monitor": self.models_monitor.get_store_settings()
        }