import json
from loguru import logger
import os

class BaseClass:
    """
    Base class for providing a serialization and deserialization mechanism.
    """
    def __init__(self, **kwargs):
        """
        Instantiates the base class with keyword arguments
        
        Args:
            kwargs (dict): Keyword arguments
        """
        self.kwargs = kwargs
        self.__dict__.update(kwargs)

    def generate_list_settings(self, list_):
        """
        Converts provided list to a normalized list that can be stored as a json object to serialize.
        
        Args:
            list_ (List): List to be converted
        Returns
            Transformed normal list
        """
        normal_list = []
        for item in list_:
            if isinstance(item, BaseClass):
                normal_list.append(item.generate_settings())
            elif isinstance(item, dict):
                normal_list.append(self.generate_kwarg_setting(item))
            elif isinstance(item, (tuple, list)):
                normal_list.append(self.generate_list_settings(item))
            else:
                normal_list.append(item)
        return normal_list

    def generate_kwarg_setting(self, kwargs):
        """
        Converts provided keyword arguments to normal kwargs in terms of serialization.

        Args:
            kwargs (dict): kwargs to be converted.
        """
        normal_kwargs = dict()
        for kwarg in kwargs:
            if isinstance(kwargs[kwarg], BaseClass):
                normal_kwargs[kwarg] = kwargs[kwarg].generate_settings()
            elif isinstance(kwargs[kwarg], (list, tuple)):
                normal_kwargs[kwarg] = self.generate_list_settings(kwargs[kwarg])
            elif isinstance(kwargs[kwarg], dict):
                normal_kwargs[kwarg] = self.generate_kwarg_setting(kwargs[kwarg])
            else:
                normal_kwargs[kwarg] = kwargs[kwarg]
        
        return normal_kwargs


    def generate_settings(self):
        """
        Generates settings for the instance of the BaseClass.

        Returns
            Settings in dictionary format.
        """
        settings = {
            "class": self.__class__.__name__, 
            **self.generate_kwarg_setting({kwarg: self.__dict__[kwarg] for kwarg in self.kwargs}), 
        }
        return settings
    
    def save(self, path):
        """
        Saves the generated settings into a JSON file at a specified path.
        
        Args:
            path (string): The file path at which the settings have to be saved.
        """
        settings = self.generate_settings()

        if os.path.dirname(path) != "":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(settings, f, indent=2)

    @classmethod
    def get_all_subclasses(cls):
        """
        Returns all subclasses of the BaseClass.
        """
        all_subclasses = []

        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(subclass.get_all_subclasses())

        return all_subclasses

    @staticmethod
    def find_class(cls_name):
        """
        Searches for a class that matches the given class name.

        Args:
            cls_name (string): Class name to be matched
        """
        for possible_cls in BaseClass.get_all_subclasses():
            if possible_cls.__name__ == cls_name:
                return possible_cls
        return None

    @staticmethod
    def load_from_list_settings(list_):
        """
        Deserializes the list saved settings to instantiate the objects.

        Args:
            list_ (List): List of saved settings
        """
        output_list = []
        for item in list_:
            if isinstance(item, dict):
                output_list.append(BaseClass.load_from_dict(item))
            elif isinstance(item, (tuple, list)):
                output_list.append(BaseClass.load_from_list_settings(item))
            else:
                output_list.append(item)

        return output_list
    
    @staticmethod
    def load_from_dict(dict_):
        """
        Deserializes the dictionary saved settings to instantiate the objects.

        Args:
            dict_ (dict): Dictionary containing saved settings
        """
        other_class = BaseClass.find_class(dict_.get("class", None))
        if other_class is not None:
            return other_class.load_from_settings(dict_)
        
        output_dict = dict()
        for key in dict_:
            if isinstance(dict_[key], dict):
                output_dict[key] = BaseClass.load_from_dict(dict_[key])
            elif isinstance(dict_[key], (tuple, list)):
                output_dict[key] = BaseClass.load_from_list_settings(dict_[key])
            else:
                output_dict[key] = dict_[key]

        return output_dict

    @staticmethod
    def load_from_settings(settings):
        """
        Deserializes the saved settings to instantiate the object.

        Args:
            settings (dict): Saved settings
        """
        cls = BaseClass.find_class(settings["class"])

        if cls is None:
            logger.error(f"Could not find class {settings['class']} when loading class.")
            return None

        kwargs = dict()
        for kwarg in settings:
            if kwarg == "class":
                continue
            if isinstance(settings[kwarg], dict):
                kwargs[kwarg] = BaseClass.load_from_dict(settings[kwarg])
            elif isinstance(settings[kwarg], (tuple, list)):
                kwargs[kwarg] = BaseClass.load_from_list_settings(settings[kwarg])
            else:
                kwargs[kwarg] = settings[kwarg]

        return cls(**kwargs)

    @classmethod
    def _load(cls, path, **kwargs):
        """
        Loads the settings from the JSON file at the specified path.
        
        Args:
            path (string): The file path from which the settings have to be loaded.
            kwargs (dict): Additional keywords arguments
        """
        with open(path, "r") as f:
            settings = json.load(f)
        for kwarg in kwargs:
            settings[kwarg] = kwargs[kwarg]
        return cls.load_from_settings(settings)

    @staticmethod
    def load(path, **kwargs):
        """
        Loads the settings of the class from the JSON file.

        Args:
            path (string): The file path from which the class settings have to be loaded.
            kwargs (dict): Additional keywords arguments
        """
        with open(path, "r") as f:
            settings = json.load(f)
        cls = BaseClass.find_class(settings["class"])
        return cls._load(path, **kwargs)

    def __str__(self) -> str:
        """
        Returns a string representation of the class object.
        """
        return f"{self.__class__.__name__}({self.kwargs})"
    
    def __eq__(self, o: object) -> bool:
        """
        Checks whether the provided object is equal to the current object.

        Args:
            o (object): Object to compare
        """
        if not isinstance(o, BaseClass):
            return False
        
        other_settings = o.generate_settings()
        settings = self.generate_settings()

        return other_settings == settings