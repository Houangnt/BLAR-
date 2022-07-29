from app.common.base import PostprocessorBase


class NOPPostprocessor(PostprocessorBase):
    def _postprocess(self, result):
        return result
