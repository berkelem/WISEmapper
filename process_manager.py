import logging
from enum import Enum
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

logger = logging.getLogger()

class RunProcess:

    def __init__(self, func, data=None, gather_items=None, parallel=True, iterate=False):
        if mpi_available and parallel:
            self.retvalue = RunParallel(func, data=data, gather_items=gather_items, iterate=iterate).retvalue
        elif mpi_available and not parallel:
            self.retvalue = RunRankZero(func, data=data, iterate=iterate).retvalue
        else:
            self.retvalue = RunLinear(func, data=data, iterate=iterate).retvalue


class Process(ABC):

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

    def __init__(self, func, **kwargs):
        super().__init__(func, **kwargs)
        self.retvalue = self.run_jobs()

    def run_jobs(self):
        job_data = self.get_job_data()
        if job_data is None:
            retvalue = self.func()
        elif self.task_index is not None:
            while self.task_index < len(job_data):
                print_progress(self.task_index, len(job_data) + 1)
                data = job_data[self.task_index]
                retvalue = self.func(data)
                self.task_index += 1

        else:
            retvalue = self.func(job_data)
        return retvalue


class RunDistributed(Process):

    def __init__(self, func, data, **kwargs):
        super().__init__(func, data, **kwargs)
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.data_sublist = data[self.rank::self.size]

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

    def run_rank_zero(self, func, data=None):
        if self.rank == 0:
            self.data = data
            job_data = self.get_job_data()
            if job_data is None:
                self.retvalue = func()
            else:
                self.retvalue = func(job_data)
        else:
            self.retvalue = None


class RunRankZero(Process):

    def __init__(self, func, **kwargs):
        super().__init__(func, **kwargs)
        # if self.rank == 0:
        #     self.retvalue = self.run_job()
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
            while self.task_index < len(job_data):
                print_progress(self.task_index + 1, len(job_data))
                data = job_data[self.task_index]
                retvalue = self.func(data)
                self.task_index += 1
        else:
            retvalue = self.func(job_data)
        return retvalue


class RunParallel(Process):

    tags = Enum('tags', ['READY', 'DONE', 'EXIT', 'START'])

    def __init__(self, func, **kwargs):
        super().__init__(func, **kwargs)
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.name = MPI.Get_processor_name()
        self.status = MPI.Status()
        if self.rank == 0:
            self.control_flow()
            self.alldata = None
            if self.gather_items:
                self.gather_to_master()
        else:
            self.run_jobs()
            if self.gather_items:
                self.gather_to_master()

    def control_flow(self):
        assert self.rank == 0
        num_workers = self.size - 1
        closed_workers = 0
        job_data = self.get_job_data()
        if job_data is None:
            self.func()

        logger.info(f"Master starting with {num_workers} workers")
        while closed_workers < num_workers:
            data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=self.status)
            source = self.status.Get_source()
            tag = self.status.Get_tag()
            if tag == type(self).tags.READY.value:
                if job_data is None:
                    self.comm.send(None, dest=source, tag=type(self).tags.EXIT.value)
                elif self.task_index < len(job_data):
                    print_progress(self.task_index, len(job_data) + 1)
                    self.comm.send(job_data[self.task_index], dest=source, tag=type(self).tags.START.value)
                    self.task_index += 1
                else:
                    self.comm.send(None, dest=source, tag=type(self).tags.EXIT.value)

            elif tag == type(self).tags.DONE.value:
                pass

            elif tag == type(self).tags.EXIT.value:
                closed_workers += 1
        return

    def run_jobs(self):
        assert self.rank > 0
        while True:
            self.comm.send(None, dest=0, tag=type(self).tags.READY.value)
            task_data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=self.status)
            tag = self.status.Get_tag()
            if tag == type(self).tags.START.value:
                result = self.func(task_data)
                self.comm.send(result, dest=0, tag=type(self).tags.DONE.value)
            elif tag == type(self).tags.EXIT.value:
                break
            elif tag == type(self).tags.SKIP.value:
                return
        self.comm.send(None, dest=0, tag=type(self).tags.EXIT.value)
        return

    def gather_to_master(self):
        self.items_to_gather = [x for x in self.gather_items]
        self.retvalue = self.comm.gather(self.items_to_gather, root=0)

