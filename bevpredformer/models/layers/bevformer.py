from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from bevpredformer.utils.debug import debug_hook


class QueryGenerator(nn.Module):
    """Interface for query generators.
    It generates the query we are looking for.
    """

    def __init__(self, query_shape, in_c, out_c):
        super().__init__()
        self._query_seq_len = query_shape
        self._output_c = out_c
        self.register_forward_hook(debug_hook)

    @property
    def query_shape(self):
        return self._query_seq_len

    @property
    def out_c(self):
        return self._output_c

    def forward(self, x) -> Dict:
        raise NotImplementedError()


class Latent2DQueryGenerator(QueryGenerator):
    def __init__(
        self,
        channels,
        bev_shape: Optional[Tuple[int, int]] = None,
        query_flat_shape: Optional[int] = None,
    ):
        if query_flat_shape is not None:
            query_shape = [query_flat_shape, 1]
        elif bev_shape is not None:
            query_shape = bev_shape
        else:
            raise ValueError("Either bev_shape or query_flat_shape must be provided")

        super().__init__(
            query_shape=query_shape,
            in_c=channels,
            out_c=channels,
        )
        self.query = nn.Parameter(0.1 * torch.randn(*query_shape, channels))

    def forward(self, bs: List[int]):
        h, w, c = self.query.shape
        query = self.query.view(*[1 for _ in range(len(bs))], h, w, c)
        query = query.repeat(*bs, 1, 1, 1)
        return query, (h, w)