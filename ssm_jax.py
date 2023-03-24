import jax.numpy as jnp
import jax.random as jr
from jax import lax
from tensorflow_probability.substrates.jax import distributions as tfd
from typing import Tuple
from dynamax.linear_gaussian_ssm import LinearGaussianSSM
from tensorflow_probability.substrates.jax.distributions import MultivariateNormalFullCovariance as MVN
from jaxtyping import Array, Float
from typing import Optional

from dynamax.utils.utils import psd_solve
from dynamax.parameters import ParameterProperties
from dynamax.types import PRNGKey, Scalar

from dynamax.linear_gaussian_ssm.inference import lgssm_posterior_sample
from dynamax.linear_gaussian_ssm.inference import ParamsLGSSM, ParamsLGSSMInitial, ParamsLGSSMDynamics, ParamsLGSSMEmissions
from dynamax.linear_gaussian_ssm.inference import PosteriorGSSMFiltered, PosteriorGSSMSmoothed

from dynamax.utils.bijectors import RealToPSDBijector


class LgssmJax(LinearGaussianSSM):
    def __init__(
            self,
            state_dim: int,
            emission_dim: int,
            input_dim: int = 0,
            has_dynamics_bias: bool = True,
            has_emissions_bias: bool = True
    ):
        super().__init__(state_dim, emission_dim, input_dim, has_dynamics_bias, has_emissions_bias)

    def initialize(
        self,
        key: PRNGKey = jr.PRNGKey(0),
        initial_mean: Optional[Float[Array, "state_dim"]]=None,
        initial_covariance=None,
        dynamics_weights=None,
        dynamics_bias=None,
        dynamics_input_weights=None,
        dynamics_covariance=None,
        emission_weights=None,
        emission_bias=None,
        emission_input_weights=None,
        emission_covariance=None
    ) -> Tuple[ParamsLGSSM, ParamsLGSSM]:
        r"""Initialize model parameters that are set to None, and their corresponding properties.

        Args:
            key: Random number key. Defaults to jr.PRNGKey(0).
            initial_mean: parameter $m$. Defaults to None.
            initial_covariance: parameter $S$. Defaults to None.
            dynamics_weights: parameter $F$. Defaults to None.
            dynamics_bias: parameter $b$. Defaults to None.
            dynamics_input_weights: parameter $B$. Defaults to None.
            dynamics_covariance: parameter $Q$. Defaults to None.
            emission_weights: parameter $H$. Defaults to None.
            emission_bias: parameter $d$. Defaults to None.
            emission_input_weights: parameter $D$. Defaults to None.
            emission_covariance: parameter $R$. Defaults to None.

        Returns:
            Tuple[ParamsLGSSM, ParamsLGSSM]: parameters and their properties.
        """

        # Arbitrary default values, for demo purposes.
        _initial_mean = jnp.zeros(self.state_dim)
        _initial_covariance = jnp.eye(self.state_dim)
        _dynamics_weights = 0.99 * jnp.eye(self.state_dim)
        _dynamics_input_weights = jnp.zeros(self.state_dim)
        _dynamics_bias = jnp.zeros((self.state_dim,)) if self.has_dynamics_bias else None
        _dynamics_covariance = 0.1 * jnp.eye(self.state_dim)
        _emission_weights = jr.normal(key, (self.emission_dim, self.state_dim))
        _emission_input_weights = jnp.zeros((self.emission_dim, self.input_dim))
        _emission_bias = jnp.zeros((self.emission_dim,)) if self.has_emissions_bias else None
        _emission_covariance = 0.1 * jnp.eye(self.emission_dim)

        # Only use the values above if the user hasn't specified their own
        default = lambda x, x0: x if x is not None else x0

        # Create nested dictionary of params
        params = ParamsLGSSM(
            initial=ParamsLGSSMInitial(
                mean=default(initial_mean, _initial_mean),
                cov=default(initial_covariance, _initial_covariance)),
            dynamics=ParamsLGSSMDynamics(
                weights=default(dynamics_weights, _dynamics_weights),
                bias=default(dynamics_bias, _dynamics_bias),
                input_weights=default(dynamics_input_weights, _dynamics_input_weights),
                cov=default(dynamics_covariance, _dynamics_covariance)),
            emissions=ParamsLGSSMEmissions(
                weights=default(emission_weights, _emission_weights),
                bias=default(emission_bias, _emission_bias),
                input_weights=default(emission_input_weights, _emission_input_weights),
                cov=default(emission_covariance, _emission_covariance))
            )

        # The keys of param_props must match those of params!
        props = ParamsLGSSM(
            initial=ParamsLGSSMInitial(
                mean=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            dynamics=ParamsLGSSMDynamics(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector())),
            emissions=ParamsLGSSMEmissions(
                weights=ParameterProperties(),
                bias=ParameterProperties(),
                input_weights=ParameterProperties(),
                cov=ParameterProperties(constrainer=RealToPSDBijector()))
            )
        return params, props

    def marginal_log_prob(
            self,
            params: ParamsLGSSM,
            emissions: Float[Array, "ntime emission_dim"],
            inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> Scalar:
        filtered_posterior = lgssm_filter_simple(params, emissions, inputs)
        return filtered_posterior.marginal_loglik

    def transition_distribution(
        self,
        params: ParamsLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.dynamics.weights @ state + jnp.diag(params.dynamics.input_weights) @ inputs
        if self.has_dynamics_bias:
            mean += params.dynamics.bias
        return MVN(mean, params.dynamics.cov)

    def emission_distribution(
        self,
        params: ParamsLGSSM,
        state: Float[Array, "state_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> tfd.Distribution:
        inputs = inputs if inputs is not None else jnp.zeros(self.input_dim)
        mean = params.emissions.weights @ state + params.emissions.input_weights @ inputs
        if self.has_emissions_bias:
            mean += params.emissions.bias
        return MVN(mean, params.emissions.cov)

    def posterior_predictive(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Tuple[Float[Array, "ntime emission_dim"], Float[Array, "ntime emission_dim"]]:
        r"""Compute marginal posterior predictive smoothing distribution for each observation.

        Args:
            params: model parameters.
            emissions: sequence of observations.
            inputs: optional sequence of inputs.

        Returns:
            :posterior predictive means $\mathbb{E}[y_{t,d} \mid y_{1:T}]$ and standard deviations $\mathrm{std}[y_{t,d} \mid y_{1:T}]$

        """
        posterior = lgssm_smoother_simple(params, emissions, inputs)
        H = params.emissions.weights
        b = params.emissions.bias
        R = params.emissions.cov
        emission_dim = R.shape[0]
        smoothed_emissions = posterior.smoothed_means @ H.T + b
        smoothed_emissions_cov = H @ posterior.smoothed_covariances @ H.T + R
        smoothed_emissions_std = jnp.sqrt(
            jnp.array([smoothed_emissions_cov[:, i, i] for i in range(emission_dim)]))
        return smoothed_emissions, smoothed_emissions_std

    def filter(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorGSSMFiltered:
        return lgssm_filter_simple(params, emissions, inputs)

    def smoother(
        self,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]] = None
    ) -> PosteriorGSSMSmoothed:
        return lgssm_smoother_simple(params, emissions, inputs)

    def posterior_sample(
        self,
        key: PRNGKey,
        params: ParamsLGSSM,
        emissions: Float[Array, "ntime emission_dim"],
        inputs: Optional[Float[Array, "ntime input_dim"]]=None
    ) -> Float[Array, "ntime state_dim"]:
        return lgssm_posterior_sample(key, params, emissions, inputs)


_get_params = lambda x, dim, t: x[t] if x.ndim == dim + 1 else x
_zeros_if_none = lambda x, shape: x if x is not None else jnp.zeros(shape)


def lgssm_filter_simple(
    params: ParamsLGSSM,
    emissions:  Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]]=None,
    nan_fill_multiplier: float=1e8
) -> PosteriorGSSMFiltered:
    r"""Run a Kalman filter to produce the marginal likelihood and filtered state estimates.

    Args:
        params: model parameters
        emissions: array of observations.
        inputs: optional array of inputs.

    Returns:
        PosteriorGSSMFiltered: filtered posterior object

    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Create a vector to replace nans in the emissions
    # if an entire time trace is nan, replace with the mean across all values
    # if the entire emission is nan, replace with 0s
    nan_fill_mean = jnp.nanmean(emissions, axis=0)
    nan_fill_mean = jnp.where(jnp.isnan(nan_fill_mean), jnp.nanmean(nan_fill_mean), nan_fill_mean)
    nan_fill_mean = jnp.where(jnp.isnan(nan_fill_mean), 0, nan_fill_mean)

    # Create a vector to set the diagonal of the covariance of the emissions to a large number
    # wherever there are NaNs in the emissions
    nan_fill_cov = jnp.nanmax(jnp.nanvar(emissions)) * nan_fill_multiplier
    nan_fill_cov = jnp.where(jnp.isnan(nan_fill_cov), nan_fill_multiplier, nan_fill_cov)

    def _step(carry, t):
        ll, pred_mean, pred_cov = carry

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics.weights, 2, t)
        B = _get_params(params.dynamics.input_weights, 2, t)
        b = _get_params(params.dynamics.bias, 1, t)
        Q = _get_params(params.dynamics.cov, 2, t)
        H = _get_params(params.emissions.weights, 2, t)
        D = _get_params(params.emissions.input_weights, 2, t)
        d = _get_params(params.emissions.bias, 1, t)
        R = _get_params(params.emissions.cov, 2, t)
        u = inputs[t]
        y = emissions[t]

        B = jnp.diag(B)

        # Find NaNs in the emissions and replace them with nan_fill_mean
        # Then, set the emission covariance to nan_fill_cov to push the filter to ignore these emissions
        nan_loc = jnp.isnan(y)
        y = jnp.where(nan_loc, nan_fill_mean, y)
        R = jnp.where(jnp.diag(nan_loc), nan_fill_cov, R)

        # Update the log likelihood
        ll += MVN(H @ pred_mean + D @ u + d, H @ pred_cov @ H.T + R).log_prob(y)

        # Condition on this emission
        filtered_mean, filtered_cov = _condition_on(pred_mean, pred_cov, H, D, d, R, u, y)

        # Predict the next state
        pred_mean, pred_cov = _predict(filtered_mean, filtered_cov, F, B, b, Q, u)

        return (ll, pred_mean, pred_cov), (filtered_mean, filtered_cov)

    # Run the Kalman filter
    carry = (0.0, params.initial.mean, params.initial.cov)
    (ll, _, _), (filtered_means, filtered_covs) = lax.scan(_step, carry, jnp.arange(num_timesteps))
    return PosteriorGSSMFiltered(marginal_loglik=ll, filtered_means=filtered_means, filtered_covariances=filtered_covs)


def _predict(m, S, F, B, b, Q, u):
    r"""Predict next mean and covariance under a linear Gaussian model.

        p(x_{t+1}) = int N(x_t \mid m, S) N(x_{t+1} \mid Fx_t + Bu + b, Q)
                    = N(x_{t+1} \mid Fm + Bu, F S F^T + Q)

    Args:
        m (D_hid,): prior mean.
        S (D_hid,D_hid): prior covariance.
        F (D_hid,D_hid): dynamics matrix.
        B (D_hid,D_in): dynamics input matrix.
        u (D_in,): inputs.
        Q (D_hid,D_hid): dynamics covariance matrix.
        b (D_hid,): dynamics bias.

    Returns:
        mu_pred (D_hid,): predicted mean.
        Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    mu_pred = F @ m + B @ u + b
    Sigma_pred = F @ S @ F.T + Q
    return mu_pred, Sigma_pred


def _condition_on(m, P, H, D, d, R, u, y):
    r"""Condition a Gaussian potential on a new linear Gaussian observation
       p(x_t \mid y_t, u_t, y_{1:t-1}, u_{1:t-1})
         propto p(x_t \mid y_{1:t-1}, u_{1:t-1}) p(y_t \mid x_t, u_t)
         = N(x_t \mid m, P) N(y_t \mid H_t x_t + D_t u_t + d_t, R_t)
         = N(x_t \mid mm, PP)
     where
         mm = m + K*(y - yhat) = mu_cond
         yhat = H*m + D*u + d
         S = (R + H * P * H')
         K = P * H' * S^{-1}
         PP = P - K S K' = Sigma_cond
     **Note! This can be done more efficiently when R is diagonal.**

    Args:
         m (D_hid,): prior mean.
         P (D_hid,D_hid): prior covariance.
         H (D_obs,D_hid): emission matrix.
         D (D_obs,D_in): emission input weights.
         u (D_in,): inputs.
         d (D_obs,): emission bias.
         R (D_obs,D_obs): emission covariance matrix.
         y (D_obs,): observation.

     Returns:
         mu_pred (D_hid,): predicted mean.
         Sigma_pred (D_hid,D_hid): predicted covariance.
    """
    # Compute the Kalman gain
    S = R + H @ P @ H.T
    K = psd_solve(S, H @ P).T
    Sigma_cond = P - K @ S @ K.T
    mu_cond = m + K @ (y - D @ u - d - H @ m)
    return mu_cond, Sigma_cond


def lgssm_smoother_simple(
    params: ParamsLGSSM,
    emissions: Float[Array, "ntime emission_dim"],
    inputs: Optional[Float[Array, "ntime input_dim"]]=None
) -> PosteriorGSSMSmoothed:
    r"""Run forward-filtering, backward-smoother to compute expectations
    under the posterior distribution on latent states. Technically, this
    implements the Rauch-Tung-Striebel (RTS) smoother.

    Args:
        params: an LGSSMParams instance (or object with the same fields)
        emissions: array of observations.
        inputs: array of inputs.

    Returns:
        PosteriorGSSMSmoothed: smoothed posterior object.

    """
    num_timesteps = len(emissions)
    inputs = jnp.zeros((num_timesteps, 0)) if inputs is None else inputs

    # Run the Kalman filter
    filtered_posterior = lgssm_filter_simple(params, emissions, inputs)
    ll, filtered_means, filtered_covs, *_ = filtered_posterior

    # Run the smoother backward in time
    def _step(carry, args):
        # Unpack the inputs
        smoothed_mean_next, smoothed_cov_next = carry
        t, filtered_mean, filtered_cov = args

        # Shorthand: get parameters and inputs for time index t
        F = _get_params(params.dynamics.weights, 2, t)
        B = _get_params(params.dynamics.input_weights, 2, t)
        b = _get_params(params.dynamics.bias, 1, t)
        Q = _get_params(params.dynamics.cov, 2, t)
        u = inputs[t]

        B = jnp.diag(B)

        # This is like the Kalman gain but in reverse
        # See Eq 8.11 of Saarka's "Bayesian Filtering and Smoothing"
        G = psd_solve(Q + F @ filtered_cov @ F.T, F @ filtered_cov).T

        # Compute the smoothed mean and covariance
        smoothed_mean = filtered_mean + G @ (smoothed_mean_next - F @ filtered_mean - B @ u - b)
        smoothed_cov = filtered_cov + G @ (smoothed_cov_next - F @ filtered_cov @ F.T - Q) @ G.T

        # Compute the smoothed expectation of x_t x_{t+1}^T
        smoothed_cross = G @ smoothed_cov_next + jnp.outer(smoothed_mean, smoothed_mean_next)

        return (smoothed_mean, smoothed_cov), (smoothed_mean, smoothed_cov, smoothed_cross)

    # Run the Kalman smoother
    init_carry = (filtered_means[-1], filtered_covs[-1])
    args = (jnp.arange(num_timesteps - 2, -1, -1), filtered_means[:-1][::-1], filtered_covs[:-1][::-1])
    _, (smoothed_means, smoothed_covs, smoothed_cross) = lax.scan(_step, init_carry, args)

    # Reverse the arrays and return
    smoothed_means = jnp.row_stack((smoothed_means[::-1], filtered_means[-1][None, ...]))
    smoothed_covs = jnp.row_stack((smoothed_covs[::-1], filtered_covs[-1][None, ...]))
    smoothed_cross = smoothed_cross[::-1]
    return PosteriorGSSMSmoothed(
        marginal_loglik=ll,
        filtered_means=filtered_means,
        filtered_covariances=filtered_covs,
        smoothed_means=smoothed_means,
        smoothed_covariances=smoothed_covs,
        smoothed_cross_covariances=smoothed_cross,
    )