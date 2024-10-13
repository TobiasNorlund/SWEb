import luigi
from pipeline.utils import ALL_CC_DUMPS

# Import all single dump tasks
from pipeline.cc_net import DownloadWET, CCNet
from pipeline.warc_processing import WARCIndex, FetchHTMLDocuments, ConvertToMarkdown
from pipeline.markdown_processing import ProcessMarkdown
from pipeline.quality_filtering import FilterDocuments

_all_tasks = [
    DownloadWET,
    CCNet,
    WARCIndex,
    FetchHTMLDocuments,
    ConvertToMarkdown,
    ProcessMarkdown,
    FilterDocuments,
]


# Define All* task classes for each single dump Task class
def _all_task_factory(cls):
    class AllTaskMetaClass(luigi.WrapperTask):
        def requires(self):
            return [cls(dump=dump) for dump in ALL_CC_DUMPS]
    AllTaskMetaClass.__name__ = f"All{cls.__name__}"
    return AllTaskMetaClass

for _task_cls in _all_tasks:
    _all_cls = _all_task_factory(_task_cls)
    globals()[_all_cls.__name__] = _all_cls
