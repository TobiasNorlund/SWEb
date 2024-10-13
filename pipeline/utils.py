import luigi
import os
from math import ceil
from dask.distributed import NannyPlugin, Nanny
from distributed.deploy.adaptive import Adaptive

from filelock import FileLock


ALL_CC_DUMPS = [
    '2013-20', '2013-48', '2014-10', '2014-15', '2014-23', '2014-35', '2014-41', '2014-42', '2014-49',
    '2014-52', '2015-06', '2015-11', '2015-14', '2015-18', '2015-22', '2015-27', '2015-32', '2015-35',
    '2015-40', '2015-48', '2016-07', '2016-18', '2016-22', '2016-26', '2016-30', '2016-36', '2016-40',
    '2016-44', '2016-50', '2017-04', '2017-09', '2017-13', '2017-17', '2017-22', '2017-26', '2017-30',
    '2017-34', '2017-39', '2017-43', '2017-47', '2017-51', '2018-05', '2018-09', '2018-13', '2018-17',
    '2018-22', '2018-26', '2018-30', '2018-34', '2018-39', '2018-43', '2018-47', '2018-51', '2019-04',
    '2019-09', '2019-13', '2019-18', '2019-22', '2019-26', '2019-30', '2019-35', '2019-39', '2019-43',
    '2019-47', '2019-51', '2020-05', '2020-10', '2020-16', '2020-24', '2020-29', '2020-34', '2020-40',
    '2020-45', '2020-50', '2021-04', '2021-10', '2021-17', '2021-21', '2021-25', '2021-31', '2021-39',
    '2021-43', '2021-49', '2022-05', '2022-21', '2022-27', '2022-33', '2022-40', '2022-49', '2023-06',
    '2023-14', '2023-23', '2023-40', '2023-50', '2024-10', '2024-18', '2024-22', '2024-26'
]


class global_conf(luigi.Config):
    account = luigi.Parameter()
    data_dir = luigi.Parameter()
    scratch_dir = luigi.Parameter()
    s3_bucket_name = luigi.Parameter()
    s3_prefix = luigi.Parameter()


GLOBAL_CONF = global_conf()


def group(iterable, group_size=None, num_groups=None):
    """
    Groups elements of `iterable` into batches. Specify either `group_size` or
    `num_groups`.
    """
    assert (group_size is None) != (num_groups is None), \
        "Error: One of group_size or num_groups must be set"
    l = len(iterable)
    if group_size is None:
        group_size = l / num_groups
    i = 0
    while round(i) < l:
        yield iterable[round(i):round(i + group_size)]
        i = i + group_size


class AssignGpuToWorker(NannyPlugin):
    """
    Nanny plugin for assigning a GPU to each worker on the same node.
    Uses a local file to increment and assign a local rank to each worker,
    and set CUDA_VISIBLE_DEVICES to that local rank.
    """
    def setup(self, nanny: Nanny):
        with FileLock("/tmp/counter.lock"):
            if not os.path.exists("/tmp/counter"):
                with open("/tmp/counter", "w") as f:
                    f.write("0")
            
            with open("/tmp/counter") as f:
                c = int(f.read())
            nanny.env["CUDA_VISIBLE_DEVICES"] = str(c)
            with open("/tmp/counter", "w") as f:
                f.write(str(c+1))


class FullNodeAdaptive(Adaptive):
    """
    Need to override adaptive target so that it never scales to a number of workers that is 
    not a multiple of workers per node. Otherwise it gets stuck in a loop killing and starting 
    new nodes indefinitely
    """
    async def target(self):
        target = await super().target()
        # Ceil to nearest multiple of 8
        return ceil(target / 8) * 8
