import math

import torch


class TraditionalInitialiser:
    """
    Initialisation for regular networks using variance scaling.
    """

    def __init__(self, gain: float = 1.):
        self.gain = gain

    def __call__(self, weight: torch.Tensor, bias: torch.Tensor):
        fan_in = torch.nn.init._calculate_correct_fan(weight, "fan_in")
        weight_dist, bias_dist = self.compute_parameters(fan_in, bias is None)
        weight_mean_sq, weight_var = weight_dist
        torch.nn.init.normal_(weight, weight_mean_sq ** .5, weight_var ** .5)
        if bias is not None:
            bias_mean, bias_var = bias_dist
            torch.nn.init.normal_(bias, bias_mean, bias_var ** .5)

    def compute_parameters(self, fan_in: int, no_bias: bool = False) -> tuple[
        tuple[float, float], tuple[float, float] | None
    ]:
        return (0., self.gain / fan_in), (0., 0.)


class ConvexBiasCorrectionInitialiser:
    """
    Initialisation method for input-convex networks that only fixes the shift.

    Notes
    -----
    This module does not allow to reproduce Figure 8 in the appendix.
    The original experiment had a bug that used convex init instead of the PyTorch default.
    As a result, using only bias initialisation is as bad as the PyTorch default!
    """

    def __init__(self, positivity, gain: float = 1.):
        # pytorch default init uses non-standard scaling
        self.gain = gain / 3.
        self.positivity = positivity

    def __call__(self, weight: torch.Tensor, bias: torch.Tensor):
        if bias is None:
            raise ValueError("Principled Initialisation for ICNNs requires bias parameter")

        fan_in = torch.nn.init._calculate_correct_fan(weight, "fan_in")
        weight_dist, bias_dist = self.compute_parameters(fan_in)
        weight_mean_sq, weight_var = weight_dist
        # pytorch default init uses uniform distribution
        offset, dist = weight_mean_sq ** .5, (3. * weight_var) ** .5
        torch.nn.init.uniform_(weight, offset - dist, offset + dist)

        bias_mean, bias_var = bias_dist
        torch.nn.init.normal_(bias, bias_mean, bias_var ** .5)

    def compute_parameters(self, fan_in: int) -> tuple[
        tuple[float, float], tuple[float, float] | None
    ]:
        weight_var = self.gain / fan_in

        # pytorch default init uses uniform distribution
        w_tmp = self.positivity((2. * torch.rand(10_000) - 1.) * (3. * weight_var) ** .5)
        shift = fan_in * w_tmp.mean() / (2 * torch.pi) ** .5
        return (0., weight_var), (-shift, 0.)


class ConvexInitialiser:
    """
    Initialisation method for input-convex networks.

    Parameters
    ----------
    var : float, optional
        The target variance fixed point.
        Should be a positive number.
    corr : float, optional
        The target correlation fixed point.
        Should be a value between -1 and 1, but typically positive.
    bias_noise : float, optional
        The fraction of variance to originate from the bias parameters.
        Should be a value between 0 and 1
    alpha : float, optional
        Scaling parameter for leaky ReLU.
        Should be a positive number.

    Examples
    --------
    Default initialisation

    >>> icnn = torch.nn.Sequential(
    ...     torch.nn.Linear(200, 400),
    ...     torch.nn.ReLU(),
    ...     ConvexLinear(400, 300),
    ... )
    >>> torch.nn.init.kaiming_uniform_(icnn[0].weight, nonlinearity="linear")
    >>> torch.nn.init.zeros_(icnn[0].bias)
    >>> convex_init = ConvexInitialiser()
    >>> w1, b1 = icnn[1].parameters()
    >>> convex_init(w1, b1)
    >>> assert torch.all(w1 >= 0) and torch.isclose(b1.var(), torch.zeros(1))

    Initialisation with random bias parameters

    >>> convex_bias_init = ConvexInitialiser(bias_noise=0.5)
    >>> convex_bias_init(w1, b1)
    >>> assert torch.all(w1 >= 0) and b1.var() > 0
    """

    @staticmethod
    @torch.no_grad()
    def init_log_normal_(weight: torch.Tensor, mean_sq: float, var: float) -> torch.Tensor:
        """
        Initialise weights with samples from a log-normal distribution.

        Parameters
        ----------
        weight : torch.Tensor
            The parameter to be initialised.
        mean_sq : float
            The squared mean of the normal distribution underlying the log-normal.
        var : float
            The variance of the normal distribution underlying the log-normal.

        Returns
        -------
        weight : torch.Tensor
            A reference to the inputs that have been modified in-place.
        """
        log_mom2 = math.log(mean_sq + var)
        log_mean = math.log(mean_sq) - log_mom2 / 2.
        log_var = log_mom2 - math.log(mean_sq)
        return torch.nn.init.normal_(weight, log_mean, log_var ** .5).exp_()

    def __init__(self, var: float = 1., corr: float = 0.5,
                 bias_noise: float = 0., alpha: float = 0.):
        self.target_var = var
        self.target_corr = corr
        self.bias_noise = bias_noise
        self.relu_scale = 2. / (1. + alpha ** 2)

    def __call__(self, weight: torch.Tensor, bias: torch.Tensor):
        if bias is None:
            raise ValueError("Principled Initialisation for ICNNs requires bias parameter")

        fan_in = torch.nn.init._calculate_correct_fan(weight, "fan_in")
        weight_dist, bias_dist = self.compute_parameters(fan_in)
        weight_mean_sq, weight_var = weight_dist
        self.init_log_normal_(weight, weight_mean_sq, weight_var)

        bias_mean, bias_var = bias_dist
        torch.nn.init.normal_(bias, bias_mean, bias_var ** .5)

    def compute_parameters(self, fan_in: int) -> tuple[
        tuple[float, float], tuple[float, float] | None
    ]:
        """
        Compute the distribution parameters for the initialisation.

        Parameters
        ----------
        fan_in : int
            Number of incoming connections.
        no_bias : bool, optional
            If `True`, computation of bias parameters is skipped.

        Returns
        -------
        (weight_mean_sq, weight_var) : tuple of 2 float
            The squared mean and variance for weight parameters.
        (bias_mean, bias_var): tuple of 2 float, optional
            The mean and variance for the bias parameters.
            If `no_bias` is `True`, `None` is returned instead.
        """
        target_mean_sq = self.target_corr / self.corr_func(fan_in)
        target_variance = self.relu_scale * (1. - self.target_corr) / fan_in

        shift = fan_in * (target_mean_sq * self.target_var / (2 * math.pi)) ** .5
        bias_var = 0.
        if self.bias_noise > 0.:
            target_variance *= (1 - self.bias_noise)
            bias_var = self.bias_noise * (1. - self.target_corr) * self.target_var

        return (target_mean_sq, target_variance), (-shift, bias_var)

    def corr_func(self, fan_in: int) -> float:
        """ Helper function for correlation (cf. $f_\mathrm{c}$, eq. 35). """
        rho = self.target_corr
        mix_mom = (1 - rho ** 2) ** .5 + rho * math.acos(-rho)
        return fan_in * (math.pi - fan_in + (fan_in - 1) * mix_mom) / (2 * math.pi)
