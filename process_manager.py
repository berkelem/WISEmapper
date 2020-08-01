"""
Module allowing simple control over actions to be taken in an MPI environment.

Main classes
------------

RunLinear : Call a function without invoking MPI distribution

RunDistributed : Call a function using MPI to distribute tasks to the number of workers defined in the sbatch command/script

RunRankZero : Call a function that only runs on the processor with rank 0, while using MPI

"""


from abc import ABC, abstractmethod
import types
import numpy as np
from myutils import print_progress
import pandas as pd


try:
    from mpi4py import MPI
    mpi_available = True
except ImportError:
    mpi_available = False


class Process(ABC):
    """Base class for module. This class handles common requirements like handling data types."""

    def __init__(self, func, data=None, gather_items=None, iterate=False):
        super().__init__()
        self.data = data
        self.func = func
        self.task_index = 0 if iterate else None
        self.retvalue = None
        self.gather_items = gather_items

    @abstractmethod
    def run_jobs(self):
        pass

    def get_job_data(self):
        """Identify the data type and extract the job data"""
        if self.data is None:
            job_data = None
        elif isinstance(self.data, str) or isinstance(self.data, int):
            job_data = self.data
        elif isinstance(self.data, types.GeneratorType):
            job_data = np.array(next(self.data))
        elif isinstance(self.data, list) or isinstance(self.data, np.ndarray):
            job_data = np.array(self.data)
        elif isinstance(self.data, pd.Series):
            job_data = self.data.values
        elif isinstance(self.data.__func__(self.data.__self__), types.GeneratorType):
            job_data = np.array(next(self.data.__func__(self.data.__self__)))
        else:
            raise TypeError(f"data is of type {type(self.data)} which is not handled by code")
        return job_data


class RunLinear(Process):
    """
    Run a function without invoking MPI.

    Parameters
    ----------
    :param func: func
        Function to run
    :param data: str or int or generator or list or numpy.ndarray or pandas.Series
        Data to be passed to function (optional)
    :param iterate: bool
        Run function iteratively on data (optional)

    Attributes
    ----------
    retvalue : Access the return value of the function

    """

    def __init__(self, func, **kwargs):
        super().__init__(func, **kwargs)
        self.retvalue = self.run_jobs()

    def run_jobs(self):
        job_data = self.get_job_data()
        if job_data is None:
            retvalue = self.func()
        elif self.task_index is not None:
            retvalue = None
            while self.task_index < len(job_data):
                print_progress(self.task_index, len(job_data) + 1)
                data = job_data[self.task_index]
                retvalue = self.func(data)
                self.task_index += 1
        else:
            retvalue = self.func(job_data)
        return retvalue


class RunDistributed(Process):
    """
        Run a function invoking MPI. The number of processes is defined elsewhere, usually in the sbatch command/script.

        Parameters
        ----------
        :param func: func
            Function to run
        :param data: str or int or generator or list or numpy.ndarray or pandas.Series
            Data to be passed to function
        :param gather_items: list
            Names of items to gather to master
        :param iterate: bool
            Run function iteratively on data (optional)


        Attributes
        ----------
        retvalue : Access the return value of the function

        """

    def __init__(self, func, data, **kwargs):
        super().__init__(func, data, **kwargs)
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.data_sublist = data[self.rank::self.size]
        self.items_to_gather = []

    def run(self):
        self.run_jobs()
        self.gather_to_master()

    def run_jobs(self):
        for i, item in enumerate(self.data_sublist):
            if self.rank == 0:
                print_progress(i+1, len(self.data_sublist))
            self.func(item)

    def gather_to_master(self):
        self.items_to_gather = [x for x in self.gather_items]
        self.retvalue = self.comm.gather(self.items_to_gather, root=0)

    # def run_rank_zero(self, func, data=None):
    #     if self.rank == 0:
    #         self.data = data
    #         job_data = self.get_job_data()
    #         if job_data is None:
    #             self.retvalue = func()
    #         else:
    #             self.retvalue = func(job_data)
    #     else:
    #         self.retvalue = None


class RunRankZero(Process):

    def __init__(self, func, **kwargs):
        super().__init__(func, **kwargs)
        if mpi_available:
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            if self.rank == 0:
                self.retvalue = self.run_jobs()
        else:
            self.retvalue = self.run_jobs()

    def run_jobs(self):
        job_data = self.get_job_data()
        if job_data is None:
            retvalue = self.func()
        elif self.task_index is not None:
            retvalue = None
            while self.task_index < len(job_data):
                print_progress(self.task_index + 1, len(job_data))
                data = job_data[self.task_index]
                retvalue = self.func(data)
                self.task_index += 1
        else:
            retvalue = self.func(job_data)
        return retvalue
