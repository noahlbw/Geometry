"""Optional exact execution backends for bounded legacy fine admission."""
from .bounded_fine_coverage import FineCoverageReader
from .fine_observer_burst import FineObserverBurstGraph
from .rival_alias_fast import retained_scores_fast
from .rival_fine_full import retained_scores


IMPLEMENTATION = 'bounded-fine-coverage-exact-execution-v1-20261006'


class FineCoverageExecution:
    def __init__(self, *, cached=True, burst=False):
        if burst and not cached:
            raise ValueError('Only the declared cached-plus-burst configuration is enabled.')
        self.cached, self.burst = cached, burst
        self.score_reader = retained_scores_fast if cached else retained_scores
        self.observer = self.graph = None

    def reader(self, observer):
        if self.observer is not None and self.observer is not observer:
            raise ValueError('An execution backend belongs to one frozen observer.')
        self.observer = observer
        if self.burst and self.graph is None:
            self.graph = FineObserverBurstGraph(observer)
        return FineCoverageReader(observer, feature_reader=self.graph)

    def report(self):
        return dict(implementation=IMPLEMENTATION, cached=self.cached, burst=self.burst,
            graph_setup_seconds={} if self.graph is None else self.graph.setup_seconds,
            graph_setup_visual_forwards=0 if self.graph is None else sum(4*n for n in self.graph.graphs),
            inference_visual_budget_unchanged=16)
