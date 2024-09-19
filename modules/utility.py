# A helper module for various sub-tasks
from time import time

def timer(func):
	"""
	Timing wrapper for a generic function.
	Prints the time spent inside the function to the output.
	"""
	def new_func(*args, **kwargs):
		start = time()
		val = func(*args,**kwargs)
		end = time()
		print(f'Time taken by {func.__name__} is {end-start:.4f} seconds')
		return val
	return new_func



class ExperimentLogger:
    def __init__(self, folder, description):
        self.folder = folder
        self.description = description
        self.objects = []
        self.logFile = folder + '/experiment_log.txt'
        with open(self.logFile, 'w') as file:
            file.write("=======================================================================\n")
            file.write("This is a short description of this experiment\n")
            file.write("=======================================================================\n")
            descriptionWithNewLine= description.replace('. ', '.\n')
            file.write(f"{descriptionWithNewLine}\n\n\n")
        
    def addSingle(self, object):
        self.objects.append(object)
        with open(self.logFile, 'a') as file:
            file.write("=======================================================================\n")
            file.write(f"Object: {object.name}\n")
            file.write("=======================================================================\n")
            for key, value in object.__dict__.items():
                file.write(f"{key} = {value}\n")
                file.write("-----------------------------------------------------------------------\n")
            file.write("\n\n")
    
    def add(self, *objects):
        for object in objects:
            self.addSingle(object)
