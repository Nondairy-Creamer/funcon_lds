import numpy as np
import scipy.stats as ss


def sampleLDSgauss(mm, nT, uu):
    # [yy,zz,yin,zin] = sampleLDSgauss(mm,nT,uu)
    #
    # Sample data from a latent LDS-Bernoulli model
    #
    # INPUTS
    # -------
    #     mm [struct] - model structure with fields
    #            .A [nz x nz] - dynamics matrix
    #            .B [nz x ns] - input matrix (optional)
    #            .C [ny x nz] - latents-to-observations matrix
    #            .D [ny x ns] - input-to-observations matrix (optional)
    #            .Q [nz x nz] - latent noise covariance
    #            .Q0 [nz x nz] - latent noise covariance for first time bin
    #            .R [ny x ny] - observation noise
    #     nT [1 x 1]   - number of time samples
    #     uu [nu x nT] - external inputs (optional)
    #
    # OUTPUTS
    # -------
    #      yy [ny x nT] - binary outputs from ny neurons
    #      zz [nz x nT] - sampled latents

    ny, nz = mm['C'].shape  # get # of neurons and # of latents

    # Initialize latents and outputs
    zz = np.zeros((nz, nT))
    yy = np.zeros((ny, nT))

    rng = np.random.default_rng(0)

    # Process inputs
    zin = np.concatenate((np.zeros((nz, 1)), mm['B'] @ uu[:, 2:]), axis=1)  # additive intput to latents
    yin = mm['D'] @ uu  # additive intput to observations

    # Sample data for first time bin
    zz[:, 1] = rng.multivariate_normal(np.zeros((1,nz)), mm['Q0']).T  # 1st latent (Note: no input in 1st time bin)
    yy[:, 1] = mm['C'] @ zz[:, 1] + yin[:, 1] + rng.multivariate_normal(np.zeros(1,ny), mm['R']).T  # 1st observation

    # Sample data for remaining bins
    for jj in range(2, nT):
        zz[:, jj] = mm['A'] @ zz[:, jj-1] + zin[:, jj] + rng.multivariate_normal(np.zeros(1, nz), mm['Q']).T  # latents
        yy[:, jj] = mm['C'] @ zz[:, jj] + yin[:, jj] + rng.multivariate_normal(np.zeros(1, ny), mm.R).T  # observations

    return yy, zz, yin, zin


def runKalmanSmooth(yy, uu, mm):
    # [zzmu,logli,zzcov,zzcov_abovediag] = runKalmanSmooth(yy,uu,mm)
    #
    # Run Kalman Filter-Smoother for latent LDS Gaussian model
    #
    # INPUTS:
    # -------
    #  yy [ny x nT] - matrix of observations
    #  uu [ns x nT] - inputs
    #  mm  [struct] - model struct with fields
    #           .A [nz x nz] - dynamics matrix
    #           .B [nz x ns] - input matrix (optional)
    #           .C [ny x nz] - latents-to-observations matrix
    #           .D [ny x ns] - input-to-observations matrix (optional)
    #           .Q [nz x nz] - latent noise covariance
    #           .R [ny x ny] - observed noise covariance
    #           .Q0 [nz x nz] - latent noise covariance for 1st time step
    #
    # OUTPUTS:
    # --------
    #    zzmu [nz x nT]      - posterior mean latents zz | yy
    #   logli [1 x 1]        - log-likelihood P( yy | ss, theta )
    #   zzcov [nz x nz x nT] - diagonal blocks of cov
    # zzcov_off [nz x nz x nT] - diagonal blocks of cov
    #
    #
    # Basic equations:
    # -----------------
    # X_t = A@X_{t-1} + w_t,    w_t ~ N(0,Q)   # latent dynamics
    # Y_t = C@X_t     + v_t,    v_t ~ N(0,R)  # observations

    # Check that dynamics matrix is stable
    # if max(abs(eig(mm.A)))>1
    #     warning('Unstable dynamics matrix: largest eigenvalue of A matrix = #f\n', max(abs(eig(mm.A))))
    #

    # Extract sizes
    nz = mm['A'].shape[0]  # number of obs and latent dimensions
    nT = yy.shape[1]  # number of time bins

    # pre-compute C'@inv(R) and C'@inv(R)@C
    CtRinv = mm['C'].T / mm['R']
    CtRinvC = CtRinv @ mm['C']

    # check if input-latent matrix is provided
    if 'B' in mm.keys() and mm['B'] is not None:
        zin = np.concatenate((np.zeros((nz, 1)), mm['B'] @ uu[:, 1:]), axis=1)  # additive intput to latents (as column vectors)
    else:
        zin = np.zeros((nz, nT))

    # check if intput-obs matrix is provided
    if 'D' in mm.keys() and mm['D'] is not None:
        yyctr = yy - (mm['D'] @ uu)  # subtract additive intput to observations
    else:
        yyctr = yy

    # Allocate storage
    zzmu = np.zeros((nz, nT))  # posterior mean E[ zz(t) | Y]
    zzcov = np.zeros((nz, nz, nT))  # marginal cov of zz(t)
    munext = np.zeros((nz, nT))  # prior mean for next step:  E[ z(t) | y(1:t-1)]
    Pnext = np.zeros((nz, nz, nT))  # prior cov for next step: cov[ z(t) | y(1:t-1)]
    logcy = np.zeros((1, nT))  # store conditionals P(y(t) | y(1:t))

    # ============================================
    # Kalman Filter (forward pass through data)
    # ============================================

    # process 1st time bin
    zzcov[:, :, 0] = np.linalg.inv(np.linalg.inv(mm['Q0']) + CtRinvC)
    zzmu[:, 0] = zzcov[:, :, 0] @ (CtRinv @ yyctr[:, 0])  # NOTE: no inputs in first time bin
    logcy[0] = ss.multivariate_normal((mm['C'] @ zin[:, 0]).T, mm['C'] @ mm['Q0'] @ mm['C'].T + mm['R']).log(yyctr[:, 0].T)

    for tt in range(1, nT + 1):
        # Step 1 (Predict):
        Pnext[:, :, tt] = mm['A'] @ zzcov[:, :, tt-1] @ mm['A'].T + mm.Q # prior cov for time bin t
        munext[:, tt] = mm['A'] @ zzmu[:, tt-1] + zin[:, tt]  # prior mean for time bin t
        # Step 2 (Update):
        zzcov[:, :, tt] = np.linalg.inv(np.linalg.inv[Pnext[:, :, tt]] + CtRinvC)   # KF cov for time bin t
        zzmu[:,tt] = zzcov[:, :, tt] @ (CtRinv @ yyctr[:, tt] + np.linalg.solve(Pnext[:, :, tt], munext[:, tt])) # KF mean

        # compute log P(y_t | y_{1:t-1})
        logcy[tt] = ss.multivariate_normal((mm['C'] @ munext[:, tt]).T, mm['C'] @ Pnext[:, :, tt] @ mm['C'].T + mm['R']).log(yyctr[:, tt].T)

    # compute marginal log-likelihood P(y | theta)
    logli = np.sum(logcy)


    # ============================================
    # Kalman Smoother (backward pass)
    # ============================================
    zzcov_offdiag1 = np.zeros((nz, nz, nT - 1))

    # Pass backwards, updating mean and covariance with info from the future
    for tt in reversed(range(1, nT)):
        Jt = (zzcov[:, :, tt] @ mm['A'].T) / Pnext[:, :, tt+1]  # matrix we need
        zzcov[:, :, tt] = zzcov[:, :, tt] + Jt @ (zzcov[:, :, tt+1] - Pnext[:, :, tt+1]) @ Jt.T # update cov
        zzmu[:, tt] = zzmu[:, tt] + Jt @ (zzmu[:, tt+1] - munext[:, tt+1])  # update mean
        zzcov_offdiag1[:, :, tt] = Jt @ zzcov[:, :, tt + 1]

    return zzmu, logli, zzcov, zzcov_offdiag1


def runEM_LDSgaussian(yy, mm, uu, optsEM):
    # [mm,logEvTrace] = runEM_LDSgaussian(yy,mm,ss,optsEM)
    #
    # Maximum likelihood fitting of LDS-Gaussian model via Expectation Maximization
    #
    # INPUTS
    # -------
    #     yy [ny x T] - Bernoulli observations- design matrix
    #
    #     mm [struct] - model structure with fields
    #            .A [nz x nz] - dynamics matrix
    #            .B [nz x ns] - input matrix (optional)
    #            .C [ny x nz] - latents-to-observations matrix
    #            .D [ny x ns] - input-to-observations matrix (optional)
    #            .Q [nz x nz] - latent noise covariance
    #            .R [ny x ny] - obs noise covariance
    #           .Q0 [nz x nz] - latent noise covariance for 1st time step
    #
    #      uu [ns x T]     - external inputs (optional)
    #
    #  optsEM [struct] - optimization params (optional)
    #       .maxiter - maximum # of iterations
    #       .dlogptol - stopping tol for change in log-likelihood
    #       .display - how often to report log-li
    #       .update  - specify which params to update during M step
    #
    # OUTPUTS
    # -------
    #          mm [struct]      - model struct with fields 'A', 'C', 'Q', and 'logEvidence'
    #  logEvTrace [1 x maxiter] - trace of log-likelihood during EM

    # Set EM optimization params if necessary
    # Set up variables for EM
    logEvTrace = np.zeros((optsEM.maxiter,1)) # trace of log-likelihood
    logpPrev = -np.inf # prev value of log-likelihood
    dlogp = np.inf # change in log-li
    jj = 0 # counter

    while (jj < optsEM['maxiter']) and (dlogp > optsEM['dlogptol']):
        jj += 1 # iteration counter

        # --- run E step  -------
        zzmu, logp, zzcov, zzcov_d1 = runKalmanSmooth(yy, uu, mm)
        logEvTrace[jj] = logp

        dlogp = logp - logpPrev # change in log-likelihood
        logpPrev = logp # update previous log-likelihood

        # Stop if LL decreased (for debugging purposes)
        if dlogp < -1e-3:
            Warning('EM iter #d (logEv = #.1f): LOG-EV DECREASED (dlogEv = #-.3g)', jj, logp, dlogp)

        # --- run M step  -------
        if uu is None:
            mm = runMstep_LDSgaussian(yy, mm, zzmu, zzcov, zzcov_d1, optsEM) # if no inputs
        else:
            mm = runMstep_LDSgaussian_wInputs(yy, uu, mm, zzmu, zzcov, zzcov_d1, optsEM)

        # ---  Display progress ----
        if np.mod(jj, optsEM.display) == 0:
            print('--- EM iter #d: logEv= #.3f ---\n', jj,logp)

    # ---- Report EM termination stats ----------
    if optsEM.display < np.inf:
        if dlogp<optsEM.dlogptol:
            print('EM finished in #d iters (dlogli=#f)\n', jj, dlogp)
            logEvTrace[jj+1:] = []
        else:
            print('EM stopped at MAXITERS=#d iters (dlogli=#f)\n',jj,dlogp)

    return mm, logEvTrace


def runMstep_LDSgaussian(yy, mm, zzmu, zzcov, zzcov_d1, optsEM):
    # mm = runMstep_LDSgaussian(yy,mm,ss,zzmu,zzcov,zzcov_d1,optsEM)
    #
    # Run M-step updates for LDS-Gaussian model
    #
    # Inputs
    # =======
    #     yy [ny x T] - Bernoulli observations- design matrix
    #     mm [struct] - model structure with fields
    #           .A [nz x nz] - dynamics matrix
    #           .B [nz x ns] - input matrix (optional)
    #           .C [ny x nz] - latents-to-observations matrix
    #           .D [ny x ns] - input-to-observations matrix (optional)
    #           .Q [nz x nz] - latent noise covariance
    #           .Q0 [ny x ny] - latent noise covariance
    #    zzmu [nz x T]     - posterior mean of latents
    #   zzcov [nz*T x nz*T] - sparse Hessian for latents around zzmap
    #
    # Output
    # =======
    #  mmnew - new model struct with updated parameters

    # Extract sizes
    nt = zzmu.shape[1]     # number of time bins

    # =============== Update dynamics parameters ==============
    if optsEM.update.Dynam:
        # Compute sufficient statistics
        Mz1 = np.sum(zzcov[:, :, :nt], dim=2) + zzmu[:, :nt] @ zzmu[:, :nt].T # E[zz@zz'] for 1 to T-1
        Mz2 = np.sum(zzcov[:, :, 2:nt+1], dim=2) + zzmu[:, 2:nt+1] @ zzmu[:, 2:nt+1].T # E[zz@zz'] for 2 to T
        Mz12 = np.sum(zzcov_d1, dim=2) + (zzmu[:, :nt] @ zzmu[:, 2:nt+1].T)   # E[zz_t@zz_{t+1}'] (above-diag)

        # update dynamics matrix A
        if optsEM['update']['A']:
            Anew = Mz12.T / Mz1
            mm['A'] = Anew

        # Update noise covariance Q
        if optsEM['update']['Q']:
            Qnew = (Mz2 + mm['A'] @ Mz1 @ mm['A'].T - mm['A'] @ Mz12 - Mz12.T @ mm['A'].T) / (nt - 1)
            Qnew = (Qnew + Qnew.T) / 2  # symmetrize (to avoid numerical issues)
            mm.Q = Qnew

    # =============== Update observation parameters ==============
    if optsEM.update.Obs:
        # Compute sufficient statistics
        if optsEM.update.Dynam:
            Mz = Mz1 + zzcov[:, :, nt] + zzmu[:, nt] @ zzmu[:, nt].T  # re-use Mz1 if possible
        else:
            Mz = np.sum(zzcov, dim=2) + zzmu @ zzmu.T # E[zz@zz']

        Mzy = zzmu @ yy.T # E[zz@yy']

        # update obs matrix C
        if optsEM.update.C:
            Cnew = Mzy.T / Mz
            mm.C = Cnew


        # update obs noise covariance R
        if optsEM['update']['R']:
            My = yy @ yy.T  # compute suff stat E[yy@yy']

            Rnew = (My + mm['C'] @ Mz @ mm['C'].T - mm['C'] @ Mzy - Mzy.T @ mm['C'].T) / nt
            # Rnew = (Rnew + Rnew')/2  # symmetrize
            mm['R'] = Rnew

    return mm


def runMstep_LDSgaussian_wInputs(yy, uu, mm, zzmu, zzcov, zzcov_d1, optsEM):
    # mm = runMstep_LDSgaussian(yy,uu,mm,zzmu,zzcov,zzcov_d1,optsEM)
    #
    # Run M-step updates for LDS-Gaussian model
    #
    # Inputs
    # =======
    #     yy [ny x T] - Bernoulli observations- design matrix
    #     uu [ns x T] - external inputs
    #     mm [struct] - model structure with fields
    #              .A [nz x nz] - dynamics matrix
    #              .B [nz x ns] - input matrix (optional)
    #              .C [ny x nz] - latents-to-observations matrix
    #              .D [ny x ns] - input-to-observations matrix (optional)
    #              .Q [nz x nz] - latent noise covariance
    #              .Q0 [ny x ny] - latent noise covariance for first latent sample
    #     zzmu [nz x T]        - posterior mean of latents
    #    zzcov [nz*T x nz*T]   -  diagonal blocks of posterior cov over latents
    # zzcov_d1 [nz*T x nz*T-1] - above-diagonal blocks of posterior covariance
    #   optsEM [struct] - optimization params (optional)
    #       .maxiter - maximum # of iterations
    #       .dlogptol - stopping tol for change in log-likelihood
    #       .display - how often to report log-li
    #       .update  - specify which params to update during M step
    #
    # Output
    # =======
    #  mmnew - new model struct with updated parameters

    # Extract sizes
    nz = mm['A'].shape[0] # number of latents
    nt = zzmu.shape[1]     # number of time bins

    # =============== Update dynamics parameters ==============
    # Compute sufficient statistics for latents
    Mz1 = np.sum(zzcov[:, :, :nt], dim=2) + zzmu[:, :nt] @ zzmu[:, :nt].T  # E[zz@zz'] for 1 to T-1
    Mz2 = np.sum(zzcov[:, :, 2:nt+1], dim=2) + zzmu[:, 2:nt+1] @ zzmu[:, 2:nt+1].T  # E[zz@zz'] for 2 to T
    Mz12 = np.sum(zzcov_d1, dim=2) + (zzmu[:,1:nt-1] @ zzmu[:, 2:nt+1].T)  # E[zz_t@zz_{t+1}'] (above-diag)

    # Compute sufficient statistics for inputs x latents
    Mu = uu[:, 2:nt+1] @ uu[:, 2:nt+1].T  # E[uu@uu'] for 2 to T
    Muz2 = uu[:, 2:nt+1] @ zzmu[:, 2:nt+1].T  # E[uu@zz'] for 2 to T
    Muz21 = uu[:, 2:nt+1] @ zzmu[:, 1:nt].T  # E[uu_t@zz_{t-1} for 2 to T

    # update dynamics matrix A & input matrix B
    if optsEM['update']['A'] and optsEM['update']['B']:
        # do a joint update for A and B
        Mlin = np.concatenate((Mz12, Muz2), axis=0) # from linear terms
        Mquad = np.concatenate((Mz1, Muz21.T, Muz21, Mu), axis=1) # from quadratic terms
        ABnew = np.linalg.solve(Mlin.T, Mquad)  # new A and B from regression
        mm['A'] = ABnew[:, :nz+1]  # new A
        mm['B'] = ABnew[:, nz+1:]  # new B

    elif optsEM['update']['A']:  # update dynamics matrix A only
        Anew = (Mz12.T - mm['B'] @ Muz21) / Mz1  # new A
        mm.A = Anew

    elif optsEM['update']['B']:  # update input matrix B only
        Bnew = (Muz2.T - mm['A'] @ Muz21.T) / Mu # new B
        mm.B = Bnew


    # Update noise covariance Q
    if optsEM['update']['Q']:
        mm['Q'] = (Mz2 + mm['A'] @ Mz1 @ mm['A'].T + mm['B'] @ Mu @ mm['B'].T
                   - mm['A'] @ Mz12 - Mz12.T @ mm['A'].T
                   - mm['B'] @ Muz2 - Muz2.T @ mm['B'].T
                   + mm['A'] @ Muz21.T @ mm['B'].T + mm['B'] @ Muz21 @ mm['A'].T) / (nt-1)

    # =============== Update observation parameters ==============
    # Compute sufficient statistics
    Mz = Mz1 + zzcov[:, :, nt] + zzmu[:, nt] @ zzmu[:, nt].T  # re-use Mz1 if possible
    Mu = Mu + uu[:, 1] @ uu[:, 1].T  # reuse Mu
    Muz = Muz2 + uu[:, 1] @ zzmu[:, 1].T # reuse Muz

    Mzy = zzmu @ yy.T  # E[zz@yy']
    Muy = uu @ yy.T   # E[uu@yy']

    # update obs matrix C & input matrix D
    if optsEM['update']['C'] and optsEM['update']['D']:
        # do a joint update to C and D
        Mlin = np.concatenate((Mzy, Muy), dim=0)  # from linear terms
        Mquad = np.concatenate((Mz, Muz.T, Muz, Mu), dim=1)  # from quadratic terms
        CDnew = np.linalg.solve(Mlin.T, Mquad)  # new A and B from regression
        mm['C'] = CDnew[:, :nz+1]  # new A
        mm['D'] = CDnew[:, nz+1:]  # new B
    elif optsEM['update']['C']:  # update C only
        Cnew = (Mzy.T - mm['D'] @ Muz) / Mz  # new A
        mm['C'] = Cnew
    elif optsEM['update']['D']:  # update D only
        Dnew = (Muy.T - mm['C'] @ Muz.T) / Mu  # new B
        mm['D'] = Dnew

    # update obs noise covariance R
    if optsEM['update']['R']:
        My = yy @ yy.T  # compute suff stat E[yy@yy']

        mm['R'] = (My + mm['C'] @ Mz @ mm['C'].T + mm['D'] @ Mu @ mm['D'].T
                   - mm['C'] @ Mzy - Mzy.T @ mm['C'].T
                   - mm['D'] @ Muy - Muy.T @ mm['D'].T
                   + mm['C'] @ Muz.T @ mm['D'].T + mm['D'] @ Muz @ mm['C'].T) / nt

    return mm


def alignLDSmodels(zz1, zztarg, mm1):
    # mmtarg = alignLDSmodels(zz1,zztarg,mm1)
    #
    # Align two LDS models by performing a linear transformation of one set of
    # latents to best match (in a least-squares sense) another set of latents.
    #
    # INPUTS
    # ------
    #    zz1 [nz x nT] - set of latents from model 1
    # zztarg [nz x nT] - set of latents from a target model
    #    mm1 [struct]  - model struct for latent LDS model, with possible fields:
    #            .A [nz x nz] - dynamics matrix
    #            .B [nz x ns] - input matrix (optional)
    #            .C [ny x nz] - latents-to-observations matrix
    #            .D [ny x ns] - input-to-observations matrix (optional)
    #            .Q [nz x nz] - latent noise covariance
    #            .R [nz x nz] - observed noise covariance
    #
    # OUTPUTS
    # ------
    #   mmtarg - new struct with params aligned as best as possible to match zztarg

    mmtarg = mm1  # initialize
    Walign = np.linalg.solve(zz1 @ zz1.T, zz1 @ zztarg.T).T  # alignment weights

    # Transform inferred params to align with true params
    if 'A' in mmtarg.keys() and mmtarg['A'] is not None:
        mmtarg['A'] = Walign @ mm1.A/Walign  # transform dynamics

    if 'B' in mmtarg.keys() and mmtarg['B'] is not None:
        mmtarg['B'] = Walign @ mm1.B     # transform input weights

    if 'C' in mmtarg.keys() and mmtarg['C'] is not None:
        mmtarg['C'] = mm1['C'] / Walign     # transform projection weights

    # transform latent noise covariances
    mmtarg['Q'] = Walign @ mm1.Q @ Walign.T  # transform latent noise covariance
    mmtarg['Q0'] = Walign @ mm1.Q0 @ Walign.T  # initial covariance

    return mmtarg














