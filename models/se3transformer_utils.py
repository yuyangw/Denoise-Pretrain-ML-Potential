
from functools import lru_cache
from typing import Dict, List
from collections import namedtuple
from itertools import product

import e3nn.o3 as o3
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.nvtx import range as nvtx_range

FiberEl = namedtuple('FiberEl', ['degree', 'channels'])

torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)


def degree_to_dim(degree: int) -> int:
    return 2 * degree + 1


@lru_cache(maxsize=None)
def get_clebsch_gordon(J: int, d_in: int, d_out: int, device) -> Tensor:
    """ Get the (cached) Q^{d_out,d_in}_J matrices from equation (8) """
    return o3.wigner_3j(J, d_in, d_out, dtype=torch.float64, device=device).permute(2, 1, 0)


@lru_cache(maxsize=None)
def get_all_clebsch_gordon(max_degree: int, device) -> List[List[Tensor]]:
    all_cb = []
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            K_Js = []
            for J in range(abs(d_in - d_out), d_in + d_out + 1):
                K_Js.append(get_clebsch_gordon(J, d_in, d_out, device))
            all_cb.append(K_Js)
    return all_cb


def get_spherical_harmonics(relative_pos: Tensor, max_degree: int) -> List[Tensor]:
    all_degrees = list(range(2 * max_degree + 1))
    sh = o3.spherical_harmonics(all_degrees, relative_pos, normalize=True)
    return torch.split(sh, [degree_to_dim(d) for d in all_degrees], dim=1)


@torch.jit.script
def get_basis_script(max_degree: int,
                     use_pad_trick: bool,
                     spherical_harmonics: List[Tensor],
                     clebsch_gordon: List[List[Tensor]],
                     amp: bool) -> Dict[str, Tensor]:
    """
    Compute pairwise bases matrices for degrees up to max_degree
    :param max_degree:            Maximum input or output degree
    :param use_pad_trick:         Pad some of the odd dimensions for a better use of Tensor Cores
    :param spherical_harmonics:   List of computed spherical harmonics
    :param clebsch_gordon:        List of computed CB-coefficients
    :param amp:                   When true, return bases in FP16 precision
    """
    basis = {}
    idx = 0
    # Double for loop instead of product() because of JIT script
    for d_in in range(max_degree + 1):
        for d_out in range(max_degree + 1):
            key = f'{d_in},{d_out}'
            K_Js = []
            for freq_idx, J in enumerate(range(abs(d_in - d_out), d_in + d_out + 1)):
                Q_J = clebsch_gordon[idx][freq_idx]
                K_Js.append(torch.einsum('n f, k l f -> n l k', spherical_harmonics[J].float(), Q_J.float()))

            basis[key] = torch.stack(K_Js, 2)  # Stack on second dim so order is n l f k
            if amp:
                basis[key] = basis[key].half()
            if use_pad_trick:
                basis[key] = F.pad(basis[key], (0, 1))  # Pad the k dimension, that can be sliced later

            idx += 1

    return basis


@torch.jit.script
def update_basis_with_fused(basis: Dict[str, Tensor],
                            max_degree: int,
                            use_pad_trick: bool,
                            fully_fused: bool) -> Dict[str, Tensor]:
    """ Update the basis dict with partially and optionally fully fused bases """
    num_edges = basis['0,0'].shape[0]
    device = basis['0,0'].device
    dtype = basis['0,0'].dtype
    sum_dim = sum([degree_to_dim(d) for d in range(max_degree + 1)])

    # Fused per output degree
    for d_out in range(max_degree + 1):
        sum_freq = sum([degree_to_dim(min(d, d_out)) for d in range(max_degree + 1)])
        basis_fused = torch.zeros(num_edges, sum_dim, sum_freq, degree_to_dim(d_out) + int(use_pad_trick),
                                  device=device, dtype=dtype)
        acc_d, acc_f = 0, 0
        for d_in in range(max_degree + 1):
            basis_fused[:, acc_d:acc_d + degree_to_dim(d_in), acc_f:acc_f + degree_to_dim(min(d_out, d_in)),
            :degree_to_dim(d_out)] = basis[f'{d_in},{d_out}'][:, :, :, :degree_to_dim(d_out)]

            acc_d += degree_to_dim(d_in)
            acc_f += degree_to_dim(min(d_out, d_in))

        basis[f'out{d_out}_fused'] = basis_fused

    # Fused per input degree
    for d_in in range(max_degree + 1):
        sum_freq = sum([degree_to_dim(min(d, d_in)) for d in range(max_degree + 1)])
        basis_fused = torch.zeros(num_edges, degree_to_dim(d_in), sum_freq, sum_dim,
                                  device=device, dtype=dtype)
        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            basis_fused[:, :, acc_f:acc_f + degree_to_dim(min(d_out, d_in)), acc_d:acc_d + degree_to_dim(d_out)] \
                = basis[f'{d_in},{d_out}'][:, :, :, :degree_to_dim(d_out)]

            acc_d += degree_to_dim(d_out)
            acc_f += degree_to_dim(min(d_out, d_in))

        basis[f'in{d_in}_fused'] = basis_fused

    if fully_fused:
        # Fully fused
        # Double sum this way because of JIT script
        sum_freq = sum([
            sum([degree_to_dim(min(d_in, d_out)) for d_in in range(max_degree + 1)]) for d_out in range(max_degree + 1)
        ])
        basis_fused = torch.zeros(num_edges, sum_dim, sum_freq, sum_dim, device=device, dtype=dtype)

        acc_d, acc_f = 0, 0
        for d_out in range(max_degree + 1):
            b = basis[f'out{d_out}_fused']
            basis_fused[:, :, acc_f:acc_f + b.shape[2], acc_d:acc_d + degree_to_dim(d_out)] = b[:, :, :,
                                                                                              :degree_to_dim(d_out)]
            acc_f += b.shape[2]
            acc_d += degree_to_dim(d_out)

        basis['fully_fused'] = basis_fused

    del basis['0,0']  # We know that the basis for l = k = 0 is filled with a constant
    return basis


def get_basis(relative_pos: Tensor,
              max_degree: int = 4,
              compute_gradients: bool = False,
              use_pad_trick: bool = False,
              amp: bool = False) -> Dict[str, Tensor]:
    with nvtx_range('spherical harmonics'):
        spherical_harmonics = get_spherical_harmonics(relative_pos, max_degree)
    with nvtx_range('CB coefficients'):
        clebsch_gordon = get_all_clebsch_gordon(max_degree, relative_pos.device)

    with torch.autograd.set_grad_enabled(compute_gradients):
        with nvtx_range('bases'):
            basis = get_basis_script(max_degree=max_degree,
                                     use_pad_trick=use_pad_trick,
                                     spherical_harmonics=spherical_harmonics,
                                     clebsch_gordon=clebsch_gordon,
                                     amp=amp)
            return basis


class Fiber(dict):
    """
    Describes the structure of some set of features.
    Features are split into types (0, 1, 2, 3, ...). A feature of type k has a dimension of 2k+1.
    Type-0 features: invariant scalars
    Type-1 features: equivariant 3D vectors
    Type-2 features: equivariant symmetric traceless matrices
    ...

    As inputs to a SE3 layer, there can be many features of the same types, and many features of different types.
    The 'multiplicity' or 'number of channels' is the number of features of a given type.
    This class puts together all the degrees and their multiplicities in order to describe
        the inputs, outputs or hidden features of SE3 layers.
    """

    def __init__(self, structure):
        if isinstance(structure, dict):
            structure = [FiberEl(int(d), int(m)) for d, m in sorted(structure.items(), key=lambda x: x[1])]
        elif not isinstance(structure[0], FiberEl):
            structure = list(map(lambda t: FiberEl(*t), sorted(structure, key=lambda x: x[1])))
        self.structure = structure
        super().__init__({d: m for d, m in self.structure})

    @property
    def degrees(self):
        return sorted([t.degree for t in self.structure])

    @property
    def channels(self):
        return [self[d] for d in self.degrees]

    @property
    def num_features(self):
        """ Size of the resulting tensor if all features were concatenated together """
        return sum(t.channels * degree_to_dim(t.degree) for t in self.structure)

    @staticmethod
    def create(num_degrees: int, num_channels: int):
        """ Create a Fiber with degrees 0..num_degrees-1, all with the same multiplicity """
        return Fiber([(degree, num_channels) for degree in range(num_degrees)])

    @staticmethod
    def from_features(feats: Dict[str, Tensor]):
        """ Infer the Fiber structure from a feature dict """
        structure = {}
        for k, v in feats.items():
            degree = int(k)
            assert len(v.shape) == 3, 'Feature shape should be (N, C, 2D+1)'
            assert v.shape[-1] == degree_to_dim(degree)
            structure[degree] = v.shape[-2]
        return Fiber(structure)

    def __getitem__(self, degree: int):
        """ fiber[degree] returns the multiplicity for this degree """
        return dict(self.structure).get(degree, 0)

    def __iter__(self):
        """ Iterate over namedtuples (degree, channels) """
        return iter(self.structure)

    def __mul__(self, other):
        """
        If other in an int, multiplies all the multiplicities by other.
        If other is a fiber, returns the cartesian product.
        """
        if isinstance(other, Fiber):
            return product(self.structure, other.structure)
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels * other for t in self.structure})

    def __add__(self, other):
        """
        If other in an int, add other to all the multiplicities.
        If other is a fiber, add the multiplicities of the fibers together.
        """
        if isinstance(other, Fiber):
            return Fiber({t.degree: t.channels + other[t.degree] for t in self.structure})
        elif isinstance(other, int):
            return Fiber({t.degree: t.channels + other for t in self.structure})

    def __repr__(self):
        return str(self.structure)

    @staticmethod
    def combine_max(f1, f2):
        """ Combine two fiber by taking the maximum multiplicity for each degree in both fibers """
        new_dict = dict(f1.structure)
        for k, m in f2.structure:
            new_dict[k] = max(new_dict.get(k, 0), m)

        return Fiber(list(new_dict.items()))

    @staticmethod
    def combine_selectively(f1, f2):
        """ Combine two fiber by taking the sum of multiplicities for each degree in the first fiber """
        # only use orders which occur in fiber f1
        new_dict = dict(f1.structure)
        for k in f1.degrees:
            if k in f2.degrees:
                new_dict[k] += f2[k]
        return Fiber(list(new_dict.items()))

    def to_attention_heads(self, tensors: Dict[str, Tensor], num_heads: int):
        # dict(N, num_channels, 2d+1) -> (N, num_heads, -1)
        fibers = [tensors[str(degree)].reshape(*tensors[str(degree)].shape[:-2], num_heads, -1) for degree in
                  self.degrees]
        fibers = torch.cat(fibers, -1)
        return fibers